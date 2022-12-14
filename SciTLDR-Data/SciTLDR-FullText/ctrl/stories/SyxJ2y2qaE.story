It is well-known that deeper neural networks are harder to train than shallower ones.

In this short paper, we use the (full) eigenvalue spectrum of the Hessian to explore how the loss landscape changes as the network gets deeper, and as residual connections are added to the architecture.

Computing a series of quantitative measures on the Hessian spectrum, we show that the Hessian eigenvalue distribution in deeper networks has substantially heavier tails (equivalently, more outlier eigenvalues), which makes the network harder to optimize with first-order methods.

We show that adding residual connections mitigates this effect substantially, suggesting a mechanism by which residual connections improve training.

Practical experience in deep learning suggests that the increased capacity that comes with deeper models can significantly improve their predictive performance.

It has also been observed that as the network becomes deeper, training becomes harder.

In convolutional neural networks (CNNs), residual connections BID5 are used to alleviate this problem.

Various explanations are provided for this phenomenon: BID6 suggests that residual connections reduce the flatness of the landscape, whereas BID3 questions this premise, noting that the extremal eigenvalues of the loss Hessian are much larger when residual connections are present: large Hessian eigenvalues indicate that the curvature of the loss is much sharper, and less flat.

In a different line of work, BID0 observes that the gradients with respect to inputs in deeper networks decorrelate with depth, and suggest that residual connections reduce the 'shattering' of the gradients.

In this paper, we explore the interaction between depth and the loss geometry.

We first establish that gradient explosion or vanishing is not responsible for the slowing down of training, as is commonly believed.

Searching for an alternative explanation, we study the Hessian eigenvalue density (using the tools introduced in BID3 to obtain estimates of the eigenvalue histogram or density).

The classical theory of strongly convex optimization tells us that optimization is slow when the spectrum simultaneously contains very small and very large eigenvalues (i.e., optimization rate is dependent on ?? = ?? max /?? min ).

Following this intuition, we focus on examining the relative spread of the Hessian eigenvalues.

In particular, we quantify the extent of the large outliers by computing some scale-invariant classical statistics of the Hessian eigenvalues, namely the skewness and kurtosis.

Finally, we observe that in comparable models with residual connections, these magnitude of these outliers is substantially mitigated.

In BID3 , it is hypothesised that batch normalization suppresses large outlier eigenvalues, thereby speeding up training; in this paper, we present evidence that residual connections speed up training through essentially the same channel.

Throughout, the dataset of interest is CIFAR-10; we describe the specific model architectures used in Appendix A.

It is well-known that deeper CNNs are harder to train than shallower ones.

We exhibit training loss curves depicting this for both residual and non-residual (we refer to these as simple) CNNs in Appendix B, at various network depths (20 and 80).

The most prevalent explanation for why very deep networks are hard to train is that the gradient explodes or vanishes as the number of layers increase BID4 ; this explanation has been infrequently challenged (Section 4.1 in BID5 ), but no definitive experiments have been shown.

We study this hypothesis in FIG0 , where we compare the gradient norms of a depth 80 residual and non-residual networks.

Two things become clear from this this plot.

Firstly, there is no exponential increase or decrease in gradient norms (i.e., we would see vastly different gradient norm scales), as hypothesised in gradient explosion explanations.

Secondly, residual connections do not consistently increase or decrease the gradient norms.

In FIG0 , 49.4% of variables have lower gradient norm in residual networks (in comparison to a baseline of non-residual networks), making the exploding/vanishing gradient explanation untenable in this case.

Let H ??? R n??n be the Hessian of the training loss function with respect to the parameters of the model: DISPLAYFORM0 where ?? ??? R n is the parameter vector, and L(??) is the training loss.

The Hessian is a measure of (local) loss curvature.

In the convex setting, optimization characteristics are largely determined by the loss curvature, we expect to be able to observe the factors slowing down the training from analyzing the Hessian spectrum along the optimization trajectory.

Let ?? 1 ??? ?? 2 ?? ?? ?? ??? ?? n be the eigenvalues of the Hessian.

The theory of convex optimization suggests that first-order methods such as SGD slow down dramatically when the relative differences among the eigenvalues of the loss Hessian are large; in particular, results from convex analysis suggest that as | ??i ??1 | becomes smaller, the optimization in the direction of the eigenvectors associated with ?? i slows down BID1 BID2 1 .

Following this intuition, when the distribution of the eigenvalues of H has heavy tails (equivalently large outliers), we expect the network to train slowly as there will be many eigenvalues where ?? i /?? 1 is small.

FIG1 shows the (smoothed) density of the eigenvalues of the Hessian for a series of simple CNNs with increasing depth.

This figure shows two prominent features of the loss Hessian:1.

Most of the eigenvalues of the Hessian are concentrated near zero.

This means that the loss surface is relatively flat, in agreement with BID8 BID3 and others.2.

As the network gets deeper, outliers appear in the spectrum of H. Moreover, the magnitude of these outliers increases with the depth of the network.

This means that as the network becomes deeper, DISPLAYFORM1 shrinks for almost all of the directions, making the training challenging.

To quantify the magnitude and extent of these outlier eigenvalues, we compute some scale-independent classical statistics of the Hessian eigenvalues.

We are primarily interested in skewness and kurtosis defined as: DISPLAYFORM2 The skewness of a distribution measures its asymmetry, and the kurtosis measures how heavy (or nonGaussian) the tails are -a heavy tailed distribution has a kurtosis greater than 3.

In our case, we compute these statistics on the Hessian eigenvalues by observing that for v ??? N (0, I n ): DISPLAYFORM3 Due to the rapid concentration of the quadratic form in high dimensions (for concrete bounds, see BID7 ) we expect extremely accurate approximation of E[?? k ] using a few i.i.d.

samples of the form v T H k v. Both skewness and kurtosis should dramatically increase as the tails of the eigenvalue density become heavier.

Figure 3 shows what happens to these metrics as we increase the depth: both the skewness and kurtosis increase dramatically as we increase the depth of the model.

In particular, note that the kurtosis is far from being a Gaussian -these distribution of eigenvalues is extremely heavy tailed.

Given that residual connections allow us to train much deeper models, we would predict that the addition of residual connections should prevent the largest eigenvalues from being so extreme.

the spectrum of the Hessian for residual networks and their corresponding simple networks (both networks are identical save for the residual connections).

We can see that adding residual connections substantially reduces the extent of the outliers in the Hessian spectrum.

More quantitatively, in Figure 3 , we can see that models with residual connections have substantially lower skewness and kurtosis than models without residual connections.

The effects are in substantial: a 90 layer model with residual connections has lower skewness and kurtosis than a non-residual model half its size.

In this paper, we have presented qualitative and quantitative evidence that depth increases outlier eigenvalues in the Hessian, and that residual connections mitigate this.

We believe that this touches upon some of the fundamental dynamics of optimizing neural networks, and that any theoretical explanation of residual connections needs to explain this.

Behrooz Ghorbani was supported by grants NSF-DMS 1418362 and NSF-DMS 1407813.

For the purposes of this short exposition, we adopt a class of networks for our study.

We consider a standard residual networks trained on CIFAR-10.

These types of networks have 6n layers of feature maps of sizes {32 ?? 32, 16 ?? 16, 8 ?? 8} (2n layers for each type) with {16, 32, 64} filters per layer.

With the addition of the input convolution layer and the final fully-connected layer, this type of network has 6n + 2 layers.

Batch-Normalization is also present in these networks.

In our experiments, when we don't include residual connections, we refer to the network 'simple-6n + 2' network and when residual connections are included, the network is referred to as ResNet-6n + 2.

We use the SGD with momentum with the same learning rate schedule to train both these networks for 100k steps.

B Deeper CNNs are harder to train; skip connections help more at depthWe observe that for small n, both simple and ResNet networks train well ( FIG6 ).

As the number of layers increase, training simple model becomes slower.

Note however that as we increase the depth, that residual

<|TLDR|>

@highlight

Network depth increases outlier eigenvalues in the Hessian. Residual connections mitigate this.