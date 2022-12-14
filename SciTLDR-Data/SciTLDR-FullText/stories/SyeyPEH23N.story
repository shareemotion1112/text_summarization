We investigate the difficulties of training sparse neural networks and make new observations about optimization dynamics and the energy landscape within the sparse regime.

Recent work of \citep{Gale2019, Liu2018} has shown that sparse ResNet-50 architectures trained on ImageNet-2012 dataset converge to solutions that are significantly worse than those found by pruning.

We show that, despite the failure of optimizers, there is a linear path with a monotonically decreasing objective from the initialization to the ``good'' solution.

Additionally, our attempts to find a decreasing objective path from ``bad'' solutions to the ``good'' ones in the sparse subspace fail.

However, if we allow the path to traverse the dense subspace, then we consistently find a path between two solutions.

These findings suggest traversing extra dimensions may be needed to escape stationary points found in the sparse subspace.

Reducing parameter footprint and inference latency of machine learning models is an active area of research, fostered by diverse applications like mobile vision and on-device intelligence.

Sparse networks, that is, neural networks in which a large subset of the model parameters are zero, have emerged as one of the leading approaches for reducing model parameter count.

It has been shown empirically that deep neural networks can achieve state-of-the-art results under high levels of sparsity BID12 BID20 BID7 , and this property has been leveraged to significantly reduce the parameter footprint and inference complexity BID15 uses the same, or even greater, computational resources compared to fully dense training, which imposes an upper limit on the size of sparse networks we can train.

Training Sparse Networks.

In the context of sparse networks, state-of-the-art results have been obtained through training densely connected networks and modifying their topology during training through a technique known as pruning BID29 BID24 BID12 .

A different approach is to reuse the sparsity pattern found through pruning and train a sparse network from scratch.

This can be done with a random initialization ("scratch") or the same initialization as the original training ("lottery").

Previous work BID7 BID18 demonstrated that both approaches achieve similar final accuracies, but lower than pruning 1 .

The difference between pruning and both approaches to training while sparse can be seen in Figure 1 .

Despite being in the same energy landscape, "scratch" and "lottery" solutions fail to match the performance of the solutions found by pruning.

Given the utility of being able to train sparse from scratch, it is critical to understand the reasons behind the failure of current techniques at training sparse neural networks.

There exists a line of work on training sparse networks BID0 BID21 BID19 BID23 which allows the connectivity pattern to change over time.

These techniques generally achieve higher accuracy compared with fixed sparse connectivity, but generally worse than pruning.

Though promising, the role of changing connections during the optimization is not clear.

While we focus on fixed sparsity pattern in this work, our results give insight into why these other approaches are more successful.

Motivated by the disparity in accuracy observed in Figure 1 , we perform a series of experiments to improve our understanding of the difficulties present in training sparse neural networks, and identify possible directions for future work.

More precisely, our main contributions are:??? A set of experiments showing that the objective function is monotonically decreasing along the straight lines that interpolate from:-the original dense initialization -the original dense initialization projected into the sparse subspace -a random initialization in the sparse subspace to the solution obtained by pruning 2 .

This demonstrates that even when the optimization process fails, there was a monotonically decreasing path to the "good" solution.??? In contrast, the linear path between the scratch and the pruned solutions depicts a high energy barrier between the two solutions.

Our attempts to find quadratic and cubic B??zier curves BID8 with a decreasing objective between the two sparse solutions fails suggesting that the optimization process gets attracted into a "bad" local minima.??? Finally, by removing the sparsity constraint from the path, we are consistently able to find decreasing objective B??zier curves between the two sparse solutions.

This result suggests that allowing for dense connectivity might be necessary and sufficient to escape the stationary point converged in the sparse subspace.

The rest of the paper is organized as follows: In ??2, we describe the experimental setup.

In ??3 we present the results from these experiments, followed by a discussion in ??4.

Training methods.

The different training strategies considered in this paper are summarized in Figure 2 .

In dense 2 sparse subspace refers to the sparsity pattern found by pruning and same in all settings.

training, we train the densely connected model and apply model pruning BID29 to find the Pruned Solution (P-S).

The other strategies start instead with a sparse connectivity pattern (represented as a binary mask) obtained from the pruned solution.

The solution obtained from the same random initialization as the pruned solution BID5 ) is denoted Lottery Solution (L-S), while the solution obtained from another random initialization ) is named Scratch Solution (S-S).

All of our experiments are based on the Resnet-50 architecture BID28 and the Imagenet-2012 dataset BID25 .

Abbreviations defined in Figure 2 below the boxes are re-used to in the remaining of the text to indicate start and end points of the interpolation experiments.

Pruning strategy.

We use magnitude based model pruning BID29 in our experiments.

This has been shown BID7 to perform as well as the more complex and computationally demanding variational dropout BID22 and 0 regularization approaches BID3 .

In our experiments, we choose the 3 top performing pruning schedules for each sparsity level using the code and checkpoints provided by BID7 .

The hyper-parameters involved in the pruning algorithm were found by grid search separately for each sparsity level.

The 80% sparse model loses almost no accuracy over the baseline, while the 98% sparse model drops to 69% top-1 accuracy (see Figure 1 -pruned).

Training details and the exact pruning schedules used in our experiments are detailed in Appendix A.Interpolation in parameter space.

Visualizing the energy landscape of neural network training is an active area of research.

BID10 measured the training loss on MNIST BID17 along the line segment between the initial point ?? s and the solution ?? e , observing a monotonically decreasing curve.

Motivated by this, we were curious if this was still true (a) for Resnet-50 on Imagenet-2012 dataset and (b) if it was still true in the sparse subspace.

We hypothesized that if (a) was true but (b) was not, then this could help explain some of the training difficulties encountered with sparse networks.

In our linear interpolation experiments, we generate networks along the segment ?? = t?? e + (1 ??? t)?? s for t ??? [???0.2, 1.2] with increments of 0.01 and evaluate them on the training set of 500k images.

Interpolated parameters include the weights, biases and trainable batch normalization parameters.

We enable training mode for batch normalization layers so that the batch statistics are used during the evaluation.

The objective is identical to the objective used during training, which includes a weight decay term scaled by 10 ???4 .We now seek to find a non-linear path between the initial point and solution using parametric B??zier curves of order n = 2 and 3.

These are curves given by the expression DISPLAYFORM0 where ?? 0 = ?? e and ?? n = ?? s .

We optimize the following, DISPLAYFORM1 using the stochastic method as proposed by BID8 with a batch size of 2048 where L(??) denotes the training loss as a function of trainable parameters.

Mirroring our original training settings, we set the weight decay coefficient to 10 ???4 .

We performed a hyper-parameter search over base learning rates (1, 10 ???1 , 10 ???2 , 10 ???3 ) and momentum coefficients (0.9, 0.95, 0.99, 0.995), obtaining similar learning curves for most of the combinations.

We choose 0.01 as the base learning rate and 0.95 as the momentum coefficient for these path finding experiments.

Our experiments highlight a gap in our understanding of energy landscape of sparse deep networks.

Why does training a sparse network from scratch gets stuck at a neighborhood of a stationary point with a significantly higher objective?

This is in contrast with recent work that has proven that such a gap does not exist for certain kinds of over-parameterized dense networks BID26 BID2 .

Since during pruning dimensions are slowly removed, we conjecture that this prevents the optimizer from getting stuck into "bad" local minima.

The failure of the optimizer is even more surprising in the light of the linear interpolation experiments of Section 3.1, which show that sparse initial points are connected to the pruning solutions through a path in which the training loss is monotonically decreasing.

In high dimensional energy landscapes, it is difficult to assess whether the training converges to a local minimum or to a higher order saddle point.

BID27 shows that the Hessian of a convolutional network trained on MNIST is degenerate and most of its eigenvalues are very close to zero indicating an extremely flat landscape at solution. (2007) 's results and argues that critical points that are far from the global minima in Gaussian fields are most likely to be saddle points.

In Section 3.2, we examine the linear interpolation between solutions and attempt to find a parametric curve between them with decreasing loss.

This is because finding a decreasing path from the high loss solution ("scratch") to the low loss solution("pruned") would demonstrate that the former solution is at a saddle point.

Linear interpolations from Initial-Point-1 (Dense), InitialPoint-1 (Sparse) and Initial-Point-2 (Sparse) to Pruned Solu- tion at different sparsity levels are shown in FIG1 respectively; all cases show monotonically decreasing curves.

The training loss represented in the y-axis consists of a cross entropy loss and an 2 regularization term.

While in FIG1 the y axis represents the full training loss, the two terms composing this loss are shown separately in Appendix C. There we observe that the sum is dominated by the cross entropy loss.

In FIG1 -left, we observe a long flat plateau followed by a sharp drop: this is unlike typical learning curves, which are steepest at the beginning and then level off.

Model pruning allows the optimizer to take the path of steepest descent while still allowing it to find a good solution as dimensions are slowly removed.

Finally, the linear interpolation from a random point sampled from the original initialization distribution ("scratch") also depicts a decreasing curve FIG1 , almost identical to the interpolations that originates from the lottery initialization FIG1 .

This brings further evidence against the "lottery ticket" initialization being special relative to other initializations.

The training loss along the linear segment and the parametric B??zier curve connecting the scratch and the pruned solutions are shown in FIG3 .

As observed by BID16 , linear interpolation FIG3 -left) depicts a barrier between solutions, as high as the values observed by randomly initialized networks.

The sparse parametric curve FIG3 -middle) found through optimization also fails at connecting the two solutions with a monotonically decreasing path (although it has much smaller loss value than the straight line).

Using a third order B??zier curve also fails to decrease the maximum loss value over the second order curve (Appendix D).

The failure of the third order curve does not prove that a path cannot be found.

However, as a second order curve was sufficient to connect solutions in dense networks BID8 , it does show that if such a path exists, then it must be significantly more complex than those necessary in dense networks.

We continue our experiments by removing the sparsity constraint from the quadratic B??zier curve and optimize over the full parameter space FIG3 .

With all dimensions unmasked, our algorithm consistently finds paths along which the objective is significantly smaller.

While this path is not strictly monotonically decreasing, this is not unexpected, given that our algorithm minimizes the integral of the objective over the interpolation segment and so monotonicity is not enforced.

We leave the exploration of monotonically decreasing paths for future work.

Our work provides insights into the dynamics of optimization in the sparse regime which we hope will guide progress towards better regularization techniques, initialization schema, and/or optimization algorithms for training sparse networks.

Training of sparse neural networks is still not fully understood from an optimization perspective.

In the sparse regime, we show that optimizers converge to stationary points with a sub-optimal generalization accuracy.

This is despite monotonically decreasing paths existing from the initial point to the pruned solution.

And despite nearly monotonically decreasing paths in the dense subspace from the "bad" local minimum to the pruned one.

Optimizers sparse networks that reach pruned accuracy levels are yet to be found.

We believe that understanding why popular optimizers used in deep learning fail in the sparse regime will yield important insights leading us towards more robust optimizers in general.

Our experiments use the code made publicly available by 3 .

The pruning algorithm uses magnitude based iterative strategy to reach to a predefined final sparsity goal over the coarse of the training.

We use a batch size of 4096 and train the network with 48000 steps (around 153 epochs).

Our learning rate starts from 0 and increases linearly towards 0.1 in 5 epoch and stays there until 50th epoch.

The learning rate is dropped by a factor of 10 afterwards at 50th, 120th and 140th epochs BID11 .

Due to high sensitivity observed, we don't prune the first convolutional layer and cap the maximum sparsity for the final fully connected layer with 80%.

Top 3 performing pruning schedules selected for each sparsity level are shared in TAB3 .

We calculate the average 1 norm of the gradient for the first setting in the table and obtain 4e-6.

As is the case for most iterative (especially stochastic) methods, our solutions do not qualify as stationary points since the gradient is never exactly zero.

By slight abuse of language we refer to stationary point as any point where the gradient is below of 10 ???5 .

Eval AccuracyRandom Sparse Initialization Figure 5 .

At the beginning of the training we randomly set a fraction of weights to zero and train the network with default parameters for 32k steps.

We observe a sudden drop only if more than 99% of the parameters are set to zero.

Initialization methods that control variance of activations during the first forward and backward-pass is known to be crucial for training deep neural networks BID9 BID13 .

However, with batch normalization BID14 and skip connections the importance of initialization is expected to be less pronounced.

sparse-init experiments shared in Figure 1 can be seen as a demonstration of such tolerance.

In sparse-init experiments we train a dense ResNet-50 but use the sparse binary mask found by pruning to set a fraction of initial weights to zero.

At all sparsity levels considered (0.8, 0.91, 0.96, 0.98), we observe that the training succeeds and reaches to final accuracy around 76% matching the performance of the original training.

Thus the initialization point alone cannot be the reason for the failure of sparse training.

To understand the extent which we are able to train sparsely initialized networks without compromising performance, we perform experiments where we randomly set a given fraction of weights to zero.

Figure 5 shows the results.

We observe no significant difference until 99.5% after which we observe a sharp drop in performance.

The initialization requires a very small number of non-zero weights to succeed.

FIG5 depicts the value of the 2 regularization term over the linear interpolations described in Section 3.1.

The curve demonstrates that the solutions found are consistently of lower weight magnitude than their initialization, and they also demonstrate that the regularization terms are a factor of ten smaller than the objective function.

Figure 7 depicts the value of the cross entropy loss over the linear interpolation described above.

FIG6 -(left) the sparse to sparse and random sparse to sparse interpolation maintain the monotonic decreasing pattern observed in the interpolations plots of the total loss FIG1 , whereas dense to sparse interpolation shows a slight increase in the objective before a rapid descent.

Dense-sparse cross entropy is increasing.

Time to drop is much less with original initializations, higher with random and least with dense.

In Section 3.2 our experiments fail to find paths along which the loss is decreasing between the "pruned" and "scratch" solutions.

Would optimizing more complex parametric curves change the result?

FIG7 depicts the objective along the third order B??zier curves.

Though the integral of the loss over the segment between solutions seems less than FIG3 -middle, we still observe a small barrier between solutions. .

Training loss along the third order (cubic) B??zier curve found between the "pruning" and "scratch" solutions.

@highlight

In this paper we highlight  the difficulty of training sparse neural networks by doing interpolation experiments in the energy landscape 