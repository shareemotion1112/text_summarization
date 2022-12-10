Pruning units in a deep network can help speed up inference and training as well as reduce the size of the model.

We show that bias propagation is a pruning technique which consistently outperforms the common approach of merely removing units,  regardless of the architecture and the dataset.

We also show how a simple adaptation to an existing scoring function allows us to select the best units to prune.

Finally,  we show that the units selected by the best performing scoring functions are somewhat consistent over the course of training, implying the dead parts of the network appear during the stages of training.

Pruning is a successful method for reducing the size of a trained neural network and accelerating inference.

Pruning consists of deleting the parts of the network whose removal least affects the network performance.

Many pruning methods proposed in the literature differ in computational cost and in effectiveness in ways that are hard to assess.

In an interesting recent work, BID3 argue for the so called "winning ticket" hypothesis.

More precisely, they train a large network after saving the random initial value of each parameter.

After training, they prune the large network to produce a smaller network with one fifth of the weights.

Setting its weights to their saved initial values and retraining achieves a performance close to that of the large trained network with a much reduced computational cost.

This result opens up a new frontier for pruning methods, where they are used to detect useless units early in the training and therefore accelerating the inference.

This contribution studies the effect of pruning methods throughout the training process.

We also present mean replacement, a unit pruning method that extends the idea of bias propagation introduced in (Ye et al., 2018) to the non-constrained training setting.

The main observations of our work can then be summarized as follows:• Regardless of the scoring function used, bias propagation reduces the pruning penalty for networks without batch normalization.• Fine-tuning the pruned network with additional training iterations reduces the bias propagation advantage but not very quickly.• Absolute valued approximation of the pruning penalty provides superior performance over the normal first order approximation.

This finding confirms the observations made by BID11 .•

Units that are selected by the best performing scoring function seem to come from a small subset of units.

This finding confirms BID3 's comments on the lottery ticket and BID2 's claims about dead units.

The rest of the paper is organized as follows.

After reviewing the related work in Section 2.

we define our pruning methods and scoring functions in Section 3.

Section 4 provides an empirical evaluation comparing various combinations of scoring functions and methods under varying pruning fractions, datasets, and models.

We briefly provide some concluding remarks and discuss future work in Section 5.

One simple and common technique to select what parts of a network to prune is to select small magnitude parameters, similar to the technique of weight decay BID5 .

BID9 and BID6 proposed using second-order saliency measures to prune trained networks with zero gradient.

With the era of deep networks, redundancy in trained networks became even more obvious and various works tackled this problem aiming to reduce the size of the network.

BID4 applied magnitude based parameter pruning on deep networks and reported around 30x compression combining various methods like weight quantization and Huffman coding.

Zhu & Gupta (2017) perform pruning during training and report similar compression rates.

BID0 focus on pruning Bayesian neural networks.

BID2 claims that the removed parameters tend to gather around specific units, so directly pruning full units might prove to be an efficient strategy.

Further, pruning units also comes with direct gains in terms of storage and speed since it meshes well with dense representations BID10 BID16 .

BID16 prune entire channels using the group lasso penalty.

BID7 observe high percentage of zero activations in deep networks with ReLU units and propose Average Percentage of Zeros as a new saliency function.

The work of BID11 focuses on iterative pruning with single unit removals in the context of transfer learning.

They compare various scoring functions and propose the absolute-valued Taylor approximation as the best performing one.

However, one practical drawback of their investigation is that they prune one unit at a time.

Once units have been selected for removal, there is still the question of how to minimize the impact of the deletion.

Most works, BID11 for instance, focus on mere removal of the units followed by retraining the network.

While retraining the full network after pruning can greatly minimize the loss in accuracy induced by the deletion, it can be computationally expensive.

Rather than recovering from the damage post-pruning, another line of research focuses on preemptively mitigating the effects.

Ye et al. (2018) propose penalizing the variance of the activations for networks using batch normalization.

They then propose replacing units with low variance with constant values using bias propagation.

Our work extends this idea of bias propagation to various other pruning methods.

BID12 suggest ablating units (which mimics removal) by replacing them with their mean activation; however, the authors report this yields inferior performance compared to simply removing the units.

BID10 propose using the l2-norm of unit activations to iteratively prune convolutional layers for VGG-16 and ResNet.

They also propose performing updates on the outgoing weights that minimize the reconstruction loss on the next layer.

Their method relies on a matrix inversion rendering it impractical for large networks.

Even with a careful unit selection, pruning can significantly damage the performance of the network.

Fine-tuning the damaged network with retraining iterations may or may not recover the full performance.

Thus our goal is to minimize the damage as much as possible at pruning time.

In this section we introduce mean replacement, a simple pruning method that significantly reduces the loss incurred by the ablation.

For a given dataset D with samples (x (i) ,y (i) ), output f (x (i) ; w) using parameters w, and loss function l(f (x (i) ; w), y (i) ; w), the loss of the optimization problem is DISPLAYFORM0 Pruning a network is often defined as setting some of its parameters to 0.

Given w, our goal then consists of finding the mask m ∈ {0, 1} d that minimizes DISPLAYFORM1 The mask m must respect some constraints.

First, even though the mask is defined at the parameter level, i.e. it has as many components as the number of parameters in the model, we are pruning units.

Hence, all parameters corresponding to the same unit must have the same value in the mask.

Second, we are interested in pruning only a limited number of units, so the number of elements set to 0 in m is constrained.

Finally, we might also want to enforce the number of units pruned at each layer, or simply prevent the pruning at some layers.

Denoting as M the set of all masks satisfying these constraints, the optimal pruning is given by DISPLAYFORM2 For the remainder of the paper, we shall assume that the number of units to remove is set for each layer independently.

This is without loss of generality and will allow us to focus on a single layer, greatly simplifying the presentation.

The complexity of solving Eq. 1 increases exponentially with the number of units to prune.

Therefore, in practice, people rank all units using a per-unit scoring function s(w; D).

Several examples of such scoring functions will be discussed in Section 3.4.

Units with the lowest score are then pruned.

This approach implicitly assumes that the scores of individual units are independent of each other.

In other words, pruning one unit is assumed to not affect the score of any other unit.

Thus, a good scoring function needs to have a small inter-unit correlation.

Once a scoring function s(w; D) has been chosen, we can define m through its elements m i : DISPLAYFORM3 where B(x, k) is the set of k elements of x with the lowest value.

In other words, we will set to 0 all the parameters belonging to units whose score is one of the k lowest, k being the number of units we wish to remove in that layer.

Pruning a fraction of the units in a particular layer can have a big impact on the network and induce a large loss penalty, that we call the pruning penalty: DISPLAYFORM4 Retraining the network might reduce the pruning penalty at the cost of additional computation.

We shall now see how adjusting the biases of the following layer can reduce the pruning penalty with low computational overhead.

In order to show this, we need to depart from our earlier definition of pruning as consisting of zeroeing a subset of the weights.

We intend to remove k units from a layer of the network in a manner that has a reasonable impact on the network performance.

This is often done by replacing these units with zeroes.

However, zero is an arbitrary choice and any constant would work.

This constant would be "propagated" by multiplying it with the outgoing weights of the layer above, which is equivalent to updating the bias of that layer with the resulting sum.

Mean replacement consists of replacing the output of pruned units by a constant that is equal to the mean of the unit outputs collected on the training samples before pruning.

A theoretical justification for that choice will be presented in Section 3.3.We will first focus on the removal of a single unit.

In a fully connected network, each unit is associated with a single activation.

However, in a convolutional layer, a unit is associated with a set of outputs, one per location.

In that case, each of these output will be replaced with the same constant.

Let a(x, p) represent the unit output for training example x at location p ∈ P. Let us randomly choose a subset 1 D s ⊂ D of examples from the training set.

We first compute the mean unit output DISPLAYFORM0 1 Usually smaller than the full training set, but big enough to get a good approximation.

(1) (2)w m 1 Figure 1 : Mean Replacement illustrated in three steps.

In step (1) the units to be pruned are selected (highlighted in red).

In step (2) mean activations are multiplied with outgoing weights.

In step (3) the product is added to the bias of corresponding units.

Mean replacement consists in replacing the pruned unit by the constantā.

This can be implemented by removing the pruned unit in the normal way -which amounts to replace its output by a zeroand folding the constantā into the bias parameter of the downstream units.

DISPLAYFORM1 where b represents the vector of the biases of the downstream units and w represents the outgoing weights of the pruned unit, that is the weights that were connecting the pruned unit to each of its downstream units before the pruning operation.

This process is illustrated in Figure 1 .We now justify our choice of constant by showing that, in the context of a quadratic loss, mean replacement is the optimal strategy.

Let us consider the linear regression setting with K samples ( DISPLAYFORM0 and parameters θ and b where h( DISPLAYFORM1 Let us write down the optimal bias for the mean square loss DISPLAYFORM2 Let us consider the case where we prune the input dimension j and denote the pruned samples with x DISPLAYFORM3 j .

The optimal bias value for this new setting is b * DISPLAYFORM4 The difference between these two optimal bias values would give us the optimal update value for the bias of the next layer after pruning, which is indeed the mean values of the pruned dimension.

DISPLAYFORM5 One can easily show that the optimal value is the sum of propagated inputs, if we prune more then one input features.

Although motivating through linear regression might not seem relevant in the deep learning case, the activations a l at layer l can be viewed as the input of the linear regression.

Each channel of the the linear function h(a l ) can be thought as a separate linear regression.

Using this observation, we can minimize the l2-norm between activations before pruning and activations after pruning, namely ||h(a l ) − h(ā l ||), fixing the weights.

BID10 take a very similar approach motivating their pruning method.

They find the optimal update without fixing the weights, requiring matrix inversion of a matrix size |D s |.

What is the optimal update for the bias in next layer?

As in the case of linear regression, we can show that the optimal update is the Mean Replacement.

Most practical pruning methods use scoring functions that assign scores to individual units.

These scoring functions attempt to assign a score to each unit such that units with small scores have the smallest loss degradation (∆L) when pruned separately.

In practice, however, scoring functions that were designed for single unit removal are used to prune k units at once.

This is valid as long as there is no cross-correlation between the scores of individual units, but this is often not the case: removing one unit usually changes the scoring distribution and possibly invalidates the previous ordering among units.

Since we aim to do unit pruning with minimal overhead, the complexity of all the scoring functions included in our experiments are linear with size of the layer or the cardinality of D s .

Throughout our experiments we compare 6 scoring functions as summarized in TAB0 ) and the features of them explained below.

Type.

There are 4 different types of scoring functions used in our experiments.

Our baseline is the random which samples scores uniformly from the range [0, 1] .

norm is the l2-norm of the unit.

One common phenomenon in training neural networks with a softmax is that the norm of the parameters tend to increase over training BID13 .

We can thus expect units that are not contributing much to the learning process to have smaller norms.

abs taylor and taylor are the Taylor approximation having the form E i∼Ds |∇ a L(a) ∆a| and E i∼Ds (∇ a L(a) ∆a) respectively.

In our experiments we use a subset D s sampled from the training set of size 1000 (Cifar-10) and 10000 (Imagenet).

taylor is the correct first order approximations for the change in the loss.

However, in practice, to our knowledge, they are not used without the absolute values.

In our experiments we confirm that they perform significantly worse compared to the other scoring functions.

We discuss the possible reasons and our observations in Section 4.3.

Approximated Penalty.

Indicates the value that is being approximated.

mean replacement indicates the pruning penalty if mean replacement is used, whereas removal indicates the penalty with regular pruning where the pruned units are just set to zero.

Mean Replaced?.This column indicates whether the pruning method itself has the bias propagation step.

We use the following experimental approach to compare various pruning strategies.

At various points during the network training, we make a copy of the network, prune a predefined fraction of its units using the chosen criterion, and measure the pruning penalty by comparing the losses measured before and after pruning.

We then resume the training process using the original copy of the network (prior to pruning).

We repeat this experiment for different convolutional networks with different sizes and depths on Cifar-10 ( BID8 ) and Imagenet-2012 (Russakovsky et al., 2015 initialized using various random seeds.

In all of our experiments we calculate the pruning penalty using subsets of sizes 1000(Cifar-10) and 10000(Imagenet) sampled from the training set.

Appendix 6.2 details the full set of experiments.

To assess the effectiveness of bias propagation across a wide variety of settings we trained various networks using the same learning rate schedule but different pruning fractions.

We pruned various combinations of layers from pruning a single layer to all layers at once.

A copy of the network was pruned every 250 steps during training, and we report the pruning penalty at these points.

FIG1 evaluates the performance of various pruning methods over the training of a five layer convolutional network and demonstrates that bias propagation reduces the pruning penalty for all pruning methods considered.

Figure 3 aggregates all such measurements plotting (x,y) pairs from each scoring function at every time step, where x-axis denotes the pruning penalty without bias propagation used and y-axis denotes the penalty with the bias propagation.

The cloud of points under the y = x line shows that bias propagation decreases the pruning penalty in almost all cases despite the variety of settings the points are sampled from (different pruning fractions, layers pruned, models trained).

One could argue that training the pruned network could quickly compensate for the damage caused by zeroing the units without bias propagation.

In other words, the networks pruned without mean replacement might end up learning the correct bias quickly through fine tuning, achieving the same loss as the network pruned with mean replacement after N fine tuning steps.

To assess this claim, we repeat our basic experiments but perform a specific number of retraining steps before measuring the post-pruning loss.

In order to eliminate the unstable effects observed during the early stages of training, in this experiment we only consider the pruning-and-retraining penalties measured after at least 25,000 training iterations on the Cifar-10 dataset.

Most of the networks we train have near zero losses by that time (see FIG1 (right)).

Figure 4 shows the scatter plots for 3 different values of fine tuning iterations.

Although the effect of Mean Replacement diminishes when we increase the number of fine tuning steps, we can still see a difference after 500 fine tuning steps, which is almost one full epoch.

This observation supports our claim that the immediate improvement on pruning penalty helps the future optimization.

In Appendix 6.5 we share the plots sampled from the other half of the results (networks pruned before the training step 25k).

And finally in Appendix 6.6 we perform some iterative pruning experiments where we see the networks pruned with various methods converge almost to the same energy level when trained long enough.

In this section we compare the performance of different scoring functions under our methodology.

To summarize the results for all experiments without losing the distance information provided by a time series plot like the one in FIG1 -(left), we use performance profiles BID1 ).

We include measurements from all Cifar-10 experiments to generate the performance profiles for all the pruning methods considered in our work.

Let us denote measurement j with tuples d (j) = (d 1 , ..., d 12 ) where d i is the pruning penalty for i'th pruning method.

Then for each such tuple we set the threshold to be t j = min(d (j) ) * τ + max(d (j) ) * (1 − τ ) for each data point j. Finally, probabilities (measured on the y-axis) for scoring function i are calculated as DISPLAYFORM0 Changing τ on the x-axis helps us to understand how close each pruning method performs to the best scoring one through the probabilistic information.

The performance profiles show several important effects: Figure 5: Performance profiles of scoring functions calculated from all experiments we ran for Cifar-10.

The y-axis denotes the probability for a particular scoring function to have a pruning penalty smaller than the threshold t i = min(∆Loss) i * τ + max(∆Loss) i * (1 − τ ) where the min and max are calculated separately among the scoring functions for each time step i. The x-axis denotes the interpolation constant τ that determines the exact threshold t i used for specific pruning measurements.

Bias propagation improves the performance of every scoring function considered.• Using Mean Replacement(lines without dashes) consistently improves performance.

This observation agrees with result in the previous section and results provided by BID11 .• ABS MRS and ABS RS have very similar performance, with the former potentially providing a small improvement over the latter.

We have observed a strong overlap between the units selected for pruning by these two methods.• The direct first order approximations of the pruning penalty, MRS and RS, perform worse than random selection.

This is very striking since it shows that the methods using pure first order approximations can have large error terms and cause serious damage to the networks.

To gain insight into this last phenomenon, we plot the output histogram of units pruned with three of our methods in FIG4 .

The corresponding pruning penalties are shown in FIG4 .

FIG4 FIG4 shows the pruning penalties for the specific experiment setting averaged over 8 seeds.

We use MEDIUM CONV network and perform pruning experiments on the second convolutional layer using a pruning fraction of 0.1.

FIG4 is the histogram of the squared activation's of the pruned units from the same experiment.

The distribution for MRS is includes many samples with high squared norm suggesting a high error term for the approximation.

FIG5 we repeat the same plot discarding the measurements taken before step 10000.

The set of units chosen by the scoring function decreases later in the training.

and therefore they provide better approximations keeping the error term of the Taylor expansion small(see Appendix 6.1 for further discussion).

We now use the same experimental setup as the FIG4 but keep track of the accumulated set of units pruned at different time steps during training.

The curves shown in FIG5 indicate which fraction of the units of a specific layer have been pruned at least once before the number of iterations specified on the horizontal axis.

These curves quickly stop increasing, indicating that the scoring functions quickly select a stable set of units for pruning.

The top curves we see in the performance profile ( Figure 5 ) appear at the bottom in FIG5 .

In other words, our best performing pruning methods selects a small subset of units for pruning relatively early during training and keep this set consistent afterwards.

This is striking because it indicates that the "winning ticket" discussed by BID3 can be identified relatively early during training.

This work presents an experimental comparison of unit pruning strategies throughout the training process.

We introduce the mean replacement approach and show that it substantially reduces the impact of the unit removal on the loss function.

We also show that fine-tuning the pruned networks does not reduce the mean replacement advantage very quickly.

We argue that direct first order approximation of the pruning penalty are poor predictors of the pruning penalty incurred by the simultaneous removal of multiple units because the neglected high order terms can become significant.

In contrast the absolute value versions of these approximations achieve the best performance.

Finally we provide some evidence showing that our best pruning methods identify a stable set of prunable units relatively early in the training process.

This last observation begs for future work.

Can we combine pruning and training in a manner that reduces the computational training cost to a quantity comparable to training the "winning ticket" network?

If we decided that we will be using Mean Replacement as our pruning method, we can define a new scoring function, i.e. the first order Taylor approximation of the pruning penalty after mean replacement.

We name this new saliency function as Mean Replacement Saliency (MRS) Let us parameterize the loss as a function with activations and write down the first order approximation of the absolute change in the loss.

DISPLAYFORM0 where DISPLAYFORM1 If we were interested in the average change in the loss we can write down the Equation 5 without the absolute values.

In other words approximations on absolute change penalizes both directions, emphasizing the change in the neural network itself rather then the loss function.

Pruning can be done at any part of the training.

Since we want to make our results as general as possible we perform experiments during the training every 250 or 10000 steps for Cifar-10 and Imagenet-2012 respectively and measure the pruning penalties.

Different settings we use in our experiments summarized in TAB2 .

We perform pruning for different sets of constraints.

First we select which layers to prune.

This can be a single layer or all layers at once.

For pruning single layers we select the first, middle and last convolutional layers and the first dense layer of each network.

We use a fraction or count to decide how many units we will be pruning at each measurement step.

If we are pruning all layers, we use the same fraction/count for all layers.

To be able to compare our results with BID11 , we also perform single unit removals.

To be able generate confidence intervals, we perform 8 experiments with each setting.

For each combination of settings in TAB2 , we pause the training every 250 iteration and perform pruning measurements on the copied model.

These measurements include calculation of scoring functions, pruning selected units, optionally doing the bias propagation and finally measuring the pruning penalty.

We perform pruning measurements for all scoring functions during the same run creating an exact copy of the model separately with and without mean propagation.

This brings us 12 pruning penalty curves for each experiment.

For each experiment we independently sample a fixed validation subset of size 1000 (cifar10) and 10000 (imagenet) from training set.

This validation set is used to calculate scoring functions, mean replacement, pruning penalty, and training loss.

We set our batch size to 64 for both datasets and perform training for 60 epochs with a learning drop of factor 10 at epoch 45.

In Section 4.2 we argued that the mean replacement helps optimization by reducing the gap between loss before and after pruning.

Particularly, we focused on the measurements done in the second half of the training.

In this section we like to share complimentary data, where instead of the second half of the training (where the networks are mostly converged), we plot the first half in FIG7 .

As expected we see more points in the negative regime, where the final loss is smaller than the loss before pruning due to the fine-tuning steps taken after pruning.

The effectiveness of our method diminishes with increased number of fine tuning steps compare to the plots shared in Section 4.2.

However we think that this comparison might not tell us a lot, due to the ongoing optimization problem and its inference with the effect of mean replacement.

As our work and experiments focus on minimizing the immediate damage on the network , many practical applications allow computational budget required for iterative pruning and fine-tuning.

Results we got in Section 4.2 and Appendix 6.5 suggests that, a network with the same sparsity but slightly worst starting point (no bias propagation) would possibly catch up the one with better starting point (mean replaced version).

In this section, we like to extend our investigation one step further and perform iterative pruning experiments with extended number of fine-tuning steps to answer whether this two starting points have different optimization paths leading to two different end points.

As number of units pruned in one pass approaches to the total number of units, all pruning methods approaches to the random scoring function.

To minimize this effect we employ iterative pruning strategy, where we prune 1% of a layer at a time and perform 100 fine tuning steps in between until the target pruning fraction is reached.

We prune all layers in the MEDIUM CONV together starting from iteration 60000.

We perform 93750 iterations(120 epoch) in total and report the average values for training loss, test loss and test accuracy over 8 different runs with 80% confidence intervals in FIG8 .

To our surprise, results in FIG8 suggests various pruning methods perform slightly better than random in our experimental setting.

We can also see the regularization effect of the pruning (test loss increases much slower than the training loss with increased target sparsity).

To investigate these results further we repeat the same experiment with VGG 11 network, 10 fine-tuning steps between pruning iterations and a starting iteration of 10000 (changing one at a time, total 7 new set of experiments).

Results from these experiments confirm the picture in FIG8 and sometimes we observe, curves running even closer to each other.

As a sanity check we use the data from Figure 4 and group the pruning penalties to compare different pruning methods.

We make 2 comparisons in one plot by changing the axis that represents our best performing method bp abs mrs keeping the two cloud of points on different sides of the diagonal.

By generating these 1-1 comparisons for increasing number of fine-tuning steps, we observe how the comparisons evolve.

If all methods would perform exactly the same, we would expect all points to converge to the diagonal and indeed, we observe such a movement in Figure 11 when we increase the number of fine tuning steps.

Blue cloud of points become almost perfectly diagonal, whereas the orange cloud(comparison against random scoring function) employs a rather slower movement supporting the picture we observe in FIG8 , particularly the difference between random and other methdods.

Fine tuning steps=500bp_abs_mrs VS abs_rs rand VS bp_abs_mrs Figure 11 : Pruning penalties from the experiments in Section 4.2 grouped by various pruning methods.

We do 2 comparisons in 1 graph by using different colours.

For a label 'Y VS X' x-axis represents pruning penalties when pruning method X is used and y-axis represents the pruning penalties when the method Y is used.

Difference among different pruning methods diminishes with increased number of fine tuning steps.

In a future work, we plan to continue investigating the discrepancy between our findings in this section with experiments made by BID11 and BID10 .

We suspect that the exact pruning strategy used, along with the hyper-parameters like learning-rate may have an import effect on the performance of pruning methods and explain the different results we observed.

<|TLDR|>

@highlight

Mean Replacement is an efficient method to improve the loss after pruning and Taylor approximation based scoring functions works better with absolute values. 

@highlight

Proposes a simple improvement to methods for unit pruning using "mean replacement"

@highlight

This paper presents a mean-replacement pruning strategy and utilizes the absolute-valued Taylor expansion as the scoring function for the pruning