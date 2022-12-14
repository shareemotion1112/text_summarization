The recent “Lottery Ticket Hypothesis” paper by Frankle & Carbin showed that a simple approach to creating sparse networks (keep the large weights) results in models that are trainable from scratch, but only when starting from the same initial weights.

The performance of these networks often exceeds the performance of the non-sparse base model, but for reasons that were not well understood.

In this paper we study the three critical components of the Lottery Ticket (LT) algorithm, showing that each may be varied significantly without impacting the overall results.

Ablating these factors leads to new insights for why LT networks perform as well as they do.

We show why setting weights to zero is important, how signs are all you need to make the re-initialized network train, and why masking behaves like training.

Finally, we discover the existence of Supermasks, or masks that can be applied to an untrained, randomly initialized network to produce a model with performance far better than chance (86% on MNIST, 41% on CIFAR-10).

Many neural networks are over-parameterized BID0 BID1 , enabling compression of each layer BID1 BID14 BID4 or of the entire network BID9 .

Some compression approaches enable more efficient computation by pruning parameters, by factorizing matrices, or via other tricks BID4 BID5 BID8 BID10 BID11 BID12 BID13 BID14 BID15 BID16 .

A recent work by Frankle & Carbin BID2 presented a simple algorithm for finding sparse subnetworks within larger networks that can meet or exceed the performance of the original network.

Their approach is as follows: after training a network, set all weights smaller than some threshold to zero BID2 , rewind the rest of the weights to their initial configuration BID3 , and then retrain the network from this starting configuration but with the zero weights frozen (not trained).

See Section S1 for a more formal description of this algorithm.

In this paper we perform ablation studies along the above three dimensions of variability, considering alternate mask criteria (Section 2), alternate mask-1 actions (Section 3), and alternate mask-0 actions (Section 4).

These studies in aggregate reveal new insights for why lottery ticket networks work as they do.

Along the way we also discover the existence of Supermasks-masks that produce above-chance performance when applied to untrained networks (Section 5).

We begin our investigation with a study of different Mask Criteria, or functions that decide which weights to keep vs. prune.

In this paper, we define the mask for each individual weight as a function of the weight's values both at initialization and after training: M (w i , w f ).

We can visualize this function as a set of decision boundaries in a 2D space as shown in FIG0 .

In BID2 , the mask criterion simply keeps weights with large final value; we refer to this as the large_final mask, M (w i , w f ) = |w f |.In addition to large_final, we also experimented with the inverse, small_final, versions that evaluate weights based on their initial magnitudes instead, large_init and small_init, versions that select for both, large_init_large_final and small_init_small_final, two mask criteria that evaluate how far weights moved, magnitude_increase and movement, and a control, random, that chooses masks randomly.

These nine masks are depicted along with their associated equations in FIG1 .

In this section and throughout the remainder of the paper, we follow the experimental framework from BID2 and perform iterative pruning experiments on a fully-connected network (FC) trained on MNIST BID7 and on three convolutional networks (Conv2, Conv4, and Conv6) trained on CIFAR-10 BID6 .

For more achitecture and training details, see Section S2.Results of these pruning experiments are shown in FIG0 .

Note that the first six criteria out of the eight form three opposing pairs; in each case, we see when one member of the pair performs better than the random baseline, the opposing member performs worse than it.

We see that the mask criterion large_final, as employed by the LT algorithm, is indeed a competitive mask criteria.

However, the magnitude_increase criterion turns out to work just as well as the large_final criterion, and in some cases significantly better.

In Section 4, we provide an explanation to why some mask criteria work well while others don't.3 Mask-1 actions: show me a signNow that we have explored various ways of choosing which weights to keep and prune, we will consider what values to set for the kept weights.

In particular, we want to explore an interesting observation in BID2 which showed that the pruned, skeletal LT networks train well when you rewind to its original initialization, but degrades in performance when you randomly reinitialize the network.

Why does reinitialization cause LT networks to train poorly?

Which components of the original initialization are important?

We evaluate a number of variants of reinitialization to investigate:• "Reinit" experiments: reinitialize kept weights based on the original initialization distribution • "Reshuffle" experiments: reinitialize while respecting the original distribution of remaining weights in that layer by reshuffling the kept weights' initial values • "Constant" experiments: reinitialize by setting mask-1 weight values to a positive or negative constant, with the constant set to be the standard deviation of each layer's original initialization.

Thus every weight on a layer becomes one of three values: −α, 0, or α.

We find that none of these three variants alone are able to train as well as the original LT network, shown as dashed lines in FIG1 .

However, all three variants work better when we ensure that the new values of the kept weights are of the same sign as their original initial values.

These are shown as solid color lines in FIG1 .

Clearly, the common factor in all working variants including the original rewind action is the sign.

As long as you keep the sign, reinitialization is not a deal breaker; in fact, even setting all kept weights to a constant value consistently performs well!

What should we do with weights that are pruned?

Typical network pruning procedures perform two actions on pruned weights: set them to zero, and freeze them in subsequent training.

However, it is unclear which of these two components leads to the increased performance in LT networks.

To separate the two factors, we run a simple experiment: we reproduce the LT iterative pruning experiments in which network weights are masked out in alternating train/mask/rewind cycles, but try an additional treatment: freeze masked weights at their initial values instead of at zero.

If pruned weights are truly unimportant, we would expect that setting them to any other sensible values, such as their original initializations, should lead to a similarly performing network.

Figure 3 shows the results for this experiment.

We find that networks perform significantly better when weights are frozen specifically at zero than at random initial values.

For these networks masked via the LT large_final criterion, zero would seem to be a particularly good value to set weights to when they had small final values.

So why does zero work better than initial values?

One hypothesis is that the mask criterion we use tends to mask to zero those weights that were headed toward zero anyway.

To test out this hypothesis, we run another experiment interpolated between the previous two: for any weight to be frozen, we freeze it to zero if it moved toward zero over the course of training, and we freeze it at its random initial value if it moved away from zero.

Results are shown in Figure 3 .

By setting only the selected subset of pruned weights to zero, we fully recover the performance of the original LT networks.

This supports our hypothesis that the benefit derived from freezing values to zero comes from the fact that those values were moving toward zero anyway.

In fact, if we apply this treatment to all weights, including the kept weights, we can outperform even the original LT networks.

The hypothesis above suggests that for certain mask criteria, like large_final, that masking is training: the masking operation tends to move weights in the direction they would have moved during training.

If so, just how powerful is this training operation?

To answer this question, we can start all the way from the beginning-not training the network at all, but simply applying a mask to the randomly initialized network.

It turns out that with a well-chosen mask, an untrained network can already attain a test accuracy far better than chance.

Although it is not entirely implausible to have better-than-chance performance since the masks are derived from the training process, the large improvement in performance is still surprising because the only transmission of information from the training back to the initial network is via a zero-one mask based on a simple criterion.

We call these masks that can produce better-than-chance accuracy without training of the underlying weights "Supermasks".

Figure 3 : Performance of various treatments of pruned weights for Conv4 on CIFAR-10.

Horizontal black line represents the performance of the original, unpruned network.

Solid blue line represents networks trained using the LT algorithm, which freeze pruned weights at zero.

Dotted blue line represents networks where pruned weights are frozen to their initial values.

Dashed grey line represents networks trained using the new proposed scheme for pruned weights: freeze pruned weights at zero if they decreased in magnitude by the end of training, otherwise freeze them at their initialization values.

Dotted grey line represents networks trained with the new proposed scheme apply to all weights by initializing kept weights to zero if they decreased in magnitude.

Performance on other models are shown in FIG4 .We turn our attention to evaluating how various mask criteria perform as Supermasks.

In addition to evaluating the mask criteria from Section 2, we define a new large_final_same_sign mask criterion based on the demonstration in Section 3 of the importance of signs and of keeping large weights.

The large_final_same_sign mask criterion selects for weights with large final magnitudes that also maintained the same sign by the end of training.

This criterion is depicted in Figure S7 .

Also included as a control is the large_final_diff_sign.

Performances of Supermasks produced by all 10 criteria are included in FIG2 , compared with two baselines: networks untrained and unmasked (untrained_baseline) and networks fully trained (trained_baseline).

By using this simple mask criterion of large_final_same_sign, we can create networks that obtain a remarkable 80% test accuracy on MNIST and 24% on CIFAR-10 without training.

Another curious observation is that if we apply the mask to a signed constant (as described in Section 3) rather than the actual initial weights, we can produce even higher test accuracy of up to 86% on MNIST and 41% on CIFAR-10!

Detailed results across network architectures, pruning percentages, and these two treaments, are shown in FIG6 .

We find it fascinating that these Supermasks exist and can be found via such simple criteria.

As an aside, they also present a method for network compression, since we only need to save a binary mask and a single random seed to reconstruct the full weights of the network.

for: Deconstructing Lottery Tickets: Zeros, Signs, and the Supermask S1 Lottery Ticket Pruning Algorithm We describe the lottery ticket algorithm in more detail: 0.

Initialize a mask m to all ones.

Randomly initialize the parameters w of a network f (x; w m) 1.

Train the parameters w of the network f (x; w m) to completion.

Denote the initial weights before training w i and the final weights after training w f .2.

Mask Criterion.

Use the mask criterion M (w i , w f ) to produce a masking score for each currently unmasked weight.

Rank the weights in each layer by their scores, set the mask value for the top p% to 1, the bottom (100 − p)% to 0, breaking ties randomly.

Here p may vary by layer, and we follow the ratios chosen in BID2 , summarized in Table S1 .

In BID2 the mask selected weights with large final value corresponding to M (w i , w f ) = |w f |.

In Section 2 we consider other mask criteria.3.

Mask-1 Action.

Take some action with the weights with mask value 1.

In BID2 these weights were reset to their initial values and marked for training in the next round.

We consider this and other mask-1 actions in Section 3.4.

Mask-0 Action.

Take some action with the weights with mask value 0.

In BID2 these weights were pruned: set to 0 and frozen during any subsequent training.

We consider this and other mask-0 actions in Section 4.

Table S1 contains the architectures used in this study, together with relevant training hyperparameters, based off of experiments in BID2 .

Table S1 : The architectures used in this paper.

Table reproduced and modified from BID2 .

Conv networks use 3x3 convolutional layers with max pooling followed by fully connected layers.

FC layer sizes are from BID7 .

Figure S2: Mask criteria studied in this section, starting with large_final that was used in BID2 .

Names we use to refer to the various methods are given along with the formula that projects each (w i , w f ) pair to a score.

Weights with the largest scores (colored regions) are kept, and weights with the smallest scores (gray regions) are pruned.

The x axis in each small figure is w i and the y axis is w f .

In two methods, α is adjusted as needed to align percentiles between w i and w f .

When masks are created, ties are broken randomly, so the random method just assigns a score of 0 to every weight.

FIG0 shows the representation of large_final mask criterion on a 2D plane.

FIG1 shows the formulation and visual depiction of all the mask criteria considered in Section 2.

Figure S3 shows the convergence speed and performance of all mask critera for FC on MNIST and Conv2, 4, 6 on CIFAR-10.

FIG2 shows the convergence speed and performance of various reinitialization methods for FC on MNIST and Conv2, 4, 6 on CIFAR-10.

FIG4 shows the performance of various treatment of pruned weights for FC on MNIST and Conv2, 4, 6 on CIFAR-10.

FIG5 illustrates why the large_final criterion biases weights that were moving toward zero during training toward zero in the mask, effectively pushing them further in the direction they were headed.

It also illustrates why the large_final criterion creates Supermasks.

Figure S7 depicts the effect of Supermasks, as well as the two additional mask criteria considered only as Supermasks.

FIG6 shows performance of various mask criteria on initial test accuracy for FC on MNIST and Conv 2, 4, 6 on CIFAR-10.

Now that we know Supermasks exist, and those derived from simple heuristics work remarkably well, we might wonder how far we can push the performance of Supermasks for a given network.

One can search, in the search space of all 2 n possible masks, where n is the number of parameters in that network.

We can also try learning it with regular optimizers.

To do that, we create a trainable mask variable for each layer while freezing all original parameters for that layer at their random initialization values.

For an original weight tensor w and a mask tensor m of the same shape, we have as the effective weight w = w i g(m), where w i denotes the initial values weights are frozen at, is element-wise multiplication and g is a point-wise function that transform a matrix of continuous values into binary values.

One example of g is (S(m)) , where S is the sigmoid function and means rounding.

Bias terms are added as usual to the product of w with the inputs as per the usual fully connected or convolutional kernels.

We train the masks with g(m) = Bern(S(m)), where Bern(p) is the bernoulli sampler with probability p.

It works slightly better than (S(m)) .

The bernoulli sampling adds some stochasticity that helps with training, mitigates the bias of all things starting at the same value, and uses in effect the expected value of S(m), which is especially useful when they are close to 0.5.By training the m matrix with SGD, we obtained up to 95.3% test accuracy on MNIST and 65.4% on CIFAR-10.

Results are shown in FIG6 , along with all the heuristic based, unlearned Supermasks.

Note that there is no straightforward way to control for the pruning percentage.

What we do is initializing m with a constant of different magnitudes, whose value nudges the network toward pruning more or less.

With this tactic we are able to produce masks with the amounts of pruning (percentages of zeros) ranging from 7% to 89%.

Further details about the training can be seen in Section S6.

TAB4 summarizes the best test accuracy obtained through different treatments.

The result shows striking improvement of learned Supermasks over heuristic based ones.

And learning Supermasks results in performance not too far from training the full network.

It suggests that a network upon initialization has already contained powerful subnetworks that work well.

Additionally, the learning of Supermask allows identifying a possibly optimal pruning rate for each layer, since each layer is free to learn the distribution of 0s in m on their own.

For instance, in BID2 the last layer of each network is designed to be pruned approximately half as much as the other layers, in our setting this ratio is automatically adjusted.

We train the networks with mask m for each layer (and all regular kernels and biases frozen) with SGD, 0.9 momentum.

The {FC, Conv2, Conv4, Conv6} networks respectively had {100, 100, 50, 20} for learning rates and trained for {2000, 2000, 1000, 800} iterations.

These hyperparameters may seem absurd, but a network of masks is quite different and cannot train well with typical learning rates.

Conv4 and Conv6 showed significant overfitting, thus we used early stopping as we are unable to use standard regularizing techniques.

For evaluation, we also use Bernoulli sampling, but average the accuracies over 10 independent samples.

For adjusting the amount pruned, we initialized m in every layer to be the same constant, which ranged from -5 to 5.

In the future it may be worth trying different initializations of m for each layer for more granular control over per-layer pruning rates.

A different method to try would be to add an L1 loss to influence layers to go toward certain values, which may alleviate the cold start problems of some networks not learning anything due to mask values starting too low (effectively having the entire network start at zero).

Figure S3 : Performance of different mask criteria for four networks at various pruning rates.

We show early stopping iteration on the left and test accuracy on the right.

Each line is a different mask criteria, with bands around magnitude_increase, large_final, movement, and random depicting the min and max over 5 runs.

Stars represent points that are significantly above all other lines at a p = 0.05 level.

large_final and magnitude_increase show the best convergence speed and accuracy, with magnitude_increase having slightly higher accuracy in Conv2 and Conv4.

As expected, criteria using small weight values consistently perform worse than random.

FIG2 : The effects of various 1-actions for the four networks and various pruning rates.

Dotted lines represent the three described methods, and solid lines are those three except with each weight having the same sign as its original initialization.

Shaded bands around notable runs depict the min and max over 5 runs.

Stars represent points that are significantly above all other lines at a p = 0.05 level.

We also include the original rewinding method and random reinitialization as baselines.

"Reshuffle, init sign" and "constant, init sign" perform similarly to the "rewind" baseline.

Figure S7 : (left) Untrained networks perform at chance (10% accuracy, for example, on the MNIST dataset as depicted), if they are randomly initialized, or randomly initialized and randomly masked.

However, applying the large_final mask improves the network accuracy beyond the chance level. (right) The large_final_same_sign mask criterion (left) that tends to produce the best Supermasks.

In contrast to the large_final mask in FIG0 , this criterion masks out the quadrants where the sign of w i and w f differ.

We include large_final_diff_sign (right) as a control.

No training is performed in any network.

Weights are frozen at either initialization or constant and various masks are applied.

Within heuristic based Supermasks (excluding learned_mask), the large_final_same_sign mask creates the highest performing Supermask by a wide margin.

Note that aside from the five independent runs performed to generate uncertainty bands for this plot, every data point on the plot is the same underlying network, just with different masks.

<|TLDR|>

@highlight

In neural network pruning, zeroing pruned weights is important, sign of initialization is key, and masking can be thought of as training.