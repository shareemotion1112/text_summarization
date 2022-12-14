We consider the question of how to assess generative adversarial networks, in particular with respect to whether or not they generalise beyond memorising the training data.

We propose a simple procedure for assessing generative adversarial network performance based on a principled consideration of what the actual goal of generalisation is.

Our approach involves using a test set to estimate the Wasserstein distance between the generative distribution produced by our procedure, and the underlying data distribution.

We use this procedure to assess the performance of several modern generative adversarial network architectures.

We find that this procedure is sensitive to the choice of ground metric on the underlying data space, and suggest a choice of ground metric that substantially improves performance.

We finally suggest that attending to the ground metric used in Wasserstein generative adversarial network training may be fruitful, and outline a concrete pathway towards doing so.

Generative adversarial networks (GANs) BID6 have attracted significant interest as a means for generative modelling.

However, recently concerns have been raised about their ability to generalise from training data and their capacity to overfit .

Moreover, techniques for evaluating the quality of GAN output are either ad hoc, lack theoretical rigor, or are not suitably objective -often times "visual inspection" of samples is the main tool of choice for the practitioner.

More fundamentally, it is sometimes unclear exactly what we want a GAN to do: what is the learning task that we are trying to achieve?In this paper, we provide a simple formulation of the GAN training framework, which consists of using a finite dataset to estimate an underlying data distribution.

The quality of GAN output is measured precisely in terms of a statistical distance D between the estimated and true distribution.

Within this context, we propose an intuitive notion of what it means for a GAN to generalise.

We also show how our notion of performance can be measured empirically for any GAN architecture when D is chosen to be a Wasserstein distance, which -unlike other methods such as the inception score BID14 ) -requires no density assumptions about the data-generating distribution.

We investigate this choice of D empirically, finding that its performance is heavily dependent on the choice of ground metric underlying the Wasserstein distance.

We suggest a novel choice of ground metric that we show performs well, and also discuss how we might otherwise use this observation to improve the design of Wasserstein GANs (WGANs) .

GANs promise a means for learning complex probability distributions in an unsupervised fashion.

In order to assess their effectiveness, we must first define precisely what we mean by this.

We seek to do so in this section, presenting a formulation of the broader goal of generative modelling that we believe is widely compatible with much present work in this area.

We also provide a natural notion of generalisation that arises in our framework.

Our setup consists of the following components.

We assume some distribution ?? on a set X .

For instance, X may be the set of 32x32 colour images, and ?? the distribution from which the CIFAR-10 dataset was sampled.

We assume that ?? is completely intractable: we do not know its density (or even if it has one), and we do not have a procedure to draw novel samples from it.

However, we do suppose that we have a fixed dataset X consisting of samples x 1 , ?? ?? ?? , x |X| iid ??? ??.

Equivalently, we have the empirical distributionX DISPLAYFORM0 where ?? denotes the Dirac mass.

Let P(X ) denote the set of probability distributions on X .

Our aim is to use X to produce a distribution in P(X ) that is as "close" as possible to ??.

We choose to measure closeness in terms of a function D : P(X ) ?? P(X ) ??? R. Usually D will be chosen to be a statistical divergence, which means that D(P, Q) ??? 0 for all P and Q, with equality iff P = Q. The task of a learning algorithm ?? in this context is then as follows:Select ??(X) ??? P(X ) such that D(??(X), ??) is as small as possible.

We believe (1) constitutes an intuitive and useful formulation of the problem of generative modelling that is largely in keeping with present research efforts.

Now, we can immediately see that one possibility is simply to choose ??(X) :=X. Moreover, in the case that D is a metric for the weak topology on P(X ) such as a Wasserstein distance, we have that D(X, ??) ??? 0 almost surely as |X| ??? ???, so that, assuming |X| is large enough, we can already expect D(X, ??) to be small.

This then suggests the following natural notion of generalisation: DISPLAYFORM1 In other words, using ?? here has actually achieved something: perhaps through some process of smoothing or interpolation, it has injected additional information intoX that has moved us closer to ?? than we were a priori.

The previous section presented (1) as a general goal of generative modelling.

In this section, we turn specifically to GANs.

We begin by providing a general model for how many of the existing varieties of GAN operate, at least ideally.

We then show how this model fits into our framework above, before considering the issue of generalisation in this context.

Most GAN algorithms in widespread use adhere to the following template: they take as input a distribution P , from which we assume we can sample, and compute (or approximate) DISPLAYFORM0 for some choices of Q ???

P(X ) and D ?? : P(X ) ?? P(X )

??? R. In other words, in the ideal case, a GAN maps P to its D ?? -projection onto Q. Note that we will not necessarily have that D ?? = D: D ?? is fixed given a particular GAN architecture, whereas the choice of D is simply a feature of our problem definition (1) and is essentially ours to make.

In practice, Q is the set of pushforward measures ?? ??? G ???1 obtained from a fixed noise distribution ?? on a noise space Z and some set G of functions G : Z ??? X .

Precisely, Q = ?? ??? G ???1 : G ??? G .

G itself usually corresponds to the set of functions realisable by some neural network architecture, and ?? is some multivariate uniform or Gaussian distribution.

However, numerous choices of D ?? have been proposed: the original GAN formulation BID6 took D ?? to be the Jenson-Shannon divergence, whereas the f -GAN BID12 generalised this to arbitrary f -divergences, and the Wasserstein GAN advocated the Wasserstein distance.

Many of the results proved in these papers involve showing that (usually under some assumptions, such as sufficient discriminator capacity) a proposed objective for G is in fact equivalent to DISPLAYFORM1 In terms of our framework in the previous section, using a GAN ?? amounts to choosing DISPLAYFORM2 We emphasise again the important distinction between D and D ?? .

In our setup, minimising D defines our ultimate goal, whereas minimising D ?? (over Q) defines how we will attempt to achieve that goal.

Even if D = D ?? , it is still at least conceivable that D(??(X), ??) might be small, and therefore this choice of ?? might be sensible.

Also note that, crucially, ?? receivesX as input rather than ?? itself.

We only have access to a fixed number of CIFAR-10 samples, for example, not an infinite stream.

Moreover, training GANs usually involves making many passes over the same dataset, so that, in effect, sampling from P will repeatedly yield the same data points.

We would not expect this to occur with nonzero probability if P = ?? for most ?? of interest.

The observation that P isX rather than ?? was also recently made by .

The authors argue that this introduces a problem for the ability of GANs to generalise, since, if D ?? is a divergence (which is almost always the case), and if Q is too big (in particular, if it is big enough that X ??? Q), then we trivially have that ??(X) =X. In other words, the GAN objective appears actively to encourage ??(X) to memorise the dataset X, and never to produce novel samples from outside of it.

The authors' proposed remedy involves trying to find a better choice of D ?? .

The problem, they argue, is that popular choices of D ?? do not satisfy the condition DISPLAYFORM3 with high probability given a modest number of samples in X. (3) They point out that this is certainly violated when D ?? is the Jensen-Shannon divergence JS, since DISPLAYFORM4 when one of P and Q is discrete and the other continuous, and give a similar result for D ?? a Wasserstein distance in the case that ?? is Gaussian.

As a solution, they introduce the neural network distance D NN defined by DISPLAYFORM5 for some choice of a class of functions F. They show that, assuming some smoothness conditions on the members of F, the choice D ?? = D NN satisfies (3), which means that if we minimise D NN (X, Q) in Q then we can be confident that the value of D NN (??, Q) is small also.

However, we do not believe that (3) is sufficient to ensure good generalisation behaviour for GANs.

What we care about ultimately is not the value of D ?? , but rather of D, and (3) invites choosing D ?? in such a way that gives no guarantees about D at all.

We see, for instance, that the degenerate choice D 0 (P, Q) := 0 trivially satisfies (3), and indeed is also a pseudometric, just like D NN .

It is therefore unclear what mathematical properties of D NN render it more suitable for estimating ?? than the obviously undesirable D 0 .

The authors do acknowledge this shortcoming of D NN , pointing out that D NN (P, Q) can be small even if P and Q are "quite different" in some sense.

The problematic consequences of the fact that P =X apply only in the case that Q is too large.

In practice, however, Q is heavily restricted, since G is restricted via a choice of neural network architecture; hence we do not know a priori whether ??(X) =X is even possible.

As such, we do not see the choice of ??(X) = ??(X) as necessarily a bad idea, and instead believe that it is an open empirical question as to how well GANs perform the task (1).

In fact, this ?? falls perfectly within the framework of minimum distance estimation BID19 BID3 , which involves estimating an underlying distribution by minimising a distance measure to a given empirical distribution.

Our goal in this section is to assess how well GANs achieve (1) by estimating D(??(X), ??) for various ?? and ??.

This raises some difficulties, given that ?? is intractable.

Our approach is to take D to be the first Wasserstein distance W d X defined by DISPLAYFORM0 where d X is a metric on X referred to as the ground metric, and ??(P, Q) denotes the set of joint distributions on the product space X ?? X with marginals P and Q. The Wasserstein distance is appealing since it is sensitive to the topology of the underlying set X , which we control by our choice of d X .

Moreover, W d X metricises weak convergence for the Wasserstein space P d X (X ) defined by DISPLAYFORM1 (see BID17 ).

Consequently, if we denote by A a set of samples DISPLAYFORM2 and by Y a set of samples (separate from X) y 1 , ?? ?? ?? , y |Y | iid ??? ??, with?? and?? the corresponding empirical distributions, then, provided DISPLAYFORM3 we have that D(??,?? ) ??? D(??(X), ??) almost surely as min {|A| , |Y |} ??? ???. Note that condition (4) holds automatically in the case that (X , d X ) is compact, since then P d X (X ) = P(X ).As such, to estimate D(??(X), ??), for D = W d X , we propose the following.

Before training, we move some of our samples from X into a testing set Y .

We next train our GAN on X, obtaining ??(X).

We then take samples A from ??(X), and obtain the estimate DISPLAYFORM4 where the left-hand side can be computed exactly by solving a linear program since both?? and Y are discrete BID16 .

We can also use the same methodology to estimate DISPLAYFORM5 , which suggests testing if DISPLAYFORM6 as a proxy for determining whether (2) holds.

A summary of our procedure is given in Algorithm 1.Algorithm 1 Procedure for testing GANs 1: Split samples from ?? into a training set X and a testing set Y 2: Compute ??(X) by training a GAN on X 3: Obtain a sample A from ??(X) DISPLAYFORM7 where the right-hand side can be computed by solving a linear program 5: Similarly, test whether DISPLAYFORM8 as a proxy for W (??(X), ??) < W (A, ??)

We applied our methodology to test two popular GAN varieties -the DCGAN BID13 and the Improved Wasserstein GAN (I-WGAN) BID8 ) -on the MNIST and CIFAR-10 datasets.

In all cases when computing the relevant Wasserstein distances, our empirical distributions consisted of 10000 samples.

We initially took our ground metric d X to be the L 2 distance.

This has the appealing property of making (X , d X ) compact when X is a space of RGB or greyscale images, therefore ensuring that (4) holds.

both datasets, W L 2 (??,?? ) decays towards an asymptote in a way that nicely corresponds to the visual quality of the samples produced.

Moreover, W L 2 (??,?? ) is much closer to W L 2 (X,?? ) towards the tail end of training for MNIST than for CIFAR-10.

This seems to reflect the fact that, visually, the eventual I-WGAN MNIST samples do seem quite close to true MNIST samples, whereas the eventual I-WGAN CIFAR-10 samples are easily identified as fake.

However, when we re-ran the same experiment using a DCGAN on CIFAR-10, we obtained the W L 2 (??,?? ) trajectories shown in Figure 2 .

Typical examples are shown in Figure 8 .

Strangely, we observe that W L 2 (??,?? ) < W L 2 (X,?? ) very early on in training -at around batch 500 -when the samples resemble the heavily blurry Figure 8b .

This raises some obvious concerns about the appropriateness of W L 2 as a metric for GAN quality.

We therefore sought to understand this strange behaviour.

Motivated by the visual blurriness of the samples in Figure 8b , we explored the effect on W L p (X,?? ) of blurring the CIFAR-10 training set X. In particular, we let X and Y each consist of 10000 distinct CIFAR-10 samples in X. We then independently convolved each channel of each image with a Gaussian kernel having standard deviation ??, obtaining a blurred dataset ?? ?? (X) and corresponding empirical distribution?? ?? (X).

The visual effect of this procedure is shown in Figure 9 in the appendix.

We then computed W L p (?? ?? (X),?? ) with ?? ranging between 0 and 10 for a variety of values of p. The results of this experiment in the case p = 2 are shown in FIG1 , and similar results were observed for other values of p: in all cases, we found that DISPLAYFORM0 whenever ?? > 0.

That is, blurring X by any amount bringsX closer to?? in W L p than not.

This occurs even though X is distributed identically to Y (both being drawn from ??), while ?? ?? (X) (presumably) is not when ?? > 0.

To remedy these issues, we sought to replace L 2 with a choice of d X that is more naturally suited to the space of images in question.

To this end we tried mapping X through a fixed pre-trained neural network ?? into a feature space Y, and then computing distances using some metric d Y on Y, rather than in X directly.

It is easily seen that, provided ?? is injective, DISPLAYFORM1 is a metric.

It also holds that, when ?? is DISPLAYFORM2 is: given a sequence x i ??? X , there exists a subsequence x i that converges in d X to some x (by compactness), so that To test the performance of W d Y ????? , we repeated the blurring experiment described above.

We took ??(x) to be the result of scaling x ??? X to size 224x224, mapping the result through a DenseNet-121 BID9 pre-trained on ImageNet BID4 , and extracting features immediately before the linear output layer.

Under the same experimental setup as above otherwise, we obtained the plot of W L 2 ????? (?? ?? (X),?? ) shown in FIG1 .

Happily, we now see that this curve increases monotonically as ?? grows in accordance with the declining visual quality of ?? ?? (X) shown in Figure 9 .

DISPLAYFORM3 Next, we computed W L 2 ????? (??,?? ) over the course of GAN training.

For the I-WGAN we obtained the results on MNIST and CIFAR-10 shown in Figure 4 ; 1 for the DCGAN on CIFAR-10, we obtained the curve shown in Figure 5 .

In all cases we see that W L 2 ????? (??,?? ) decreases monotonically towards an asymptote in a way that accurately summarises the visual quality of the samples throughout the training run.

Moreover, there is always a large gap between the eventual value of W L 2 ????? (??,?? ) and W L 2 ????? (X,?? ), which reflects the fact that the GAN samples are still visually distinguishable from real ?? samples.

In particular, we see an improvement in this respect for the I-WGAN on MNIST: in Figure 1 the asymptotic value of W L 2 (??,?? ) for MNIST was barely discernible from W L 2 (X,?? ), despite the fact that it is still quite easy to tell real samples from generated ones (see e.g. the various mistakes present in Figure 6 ).

We believe our work reveals two promising avenues of future inquiry.

First, we suggest that W L p ????? is an appealing choice of D, both due to its nice theoretical properties -it metricises weak convergence, and does not require us to make any density assumptions about ?? -and due to its sound empirical performance demonstrated above.

It would be very interesting to use this D to produce a systematic and objective comparison of the performance of all current major GAN implementations, and indeed to use this as a metric for guiding future GAN design.

We also view the test (5) as potentially useful for determining whether our algorithms are overfitting.

This would be particularly so if applied via a cross-validation procedure: if we consistently observe that (5) holds when training a GAN according to many different X and Y partitions of our total ?? samples, then it seems reasonable to infer that ??(X) has indeed learnt something useful about ??.

We also believe that the empirical inadequacy of W L 2 that we observed suggests a path towards a better WGAN architecture.

At present, WGAN implementations implicitly use W L 2 for their choice of D ?? .

We suspect that altering this to our suggested W L 2 ????? may yield better quality samples.

We briefly give here one possible way to do so that is largely compatible with existing WGAN setups.

In particular, following , we take DISPLAYFORM0 for a class F of functions f : X ??? R that are all (L 2 ??? ??, d R )-Lipschitz for some fixed Lipschitz constant K. Here d R denotes the usual distance on R. To optimise over such an F in practice, we can require our discriminator f : X ??? R to have the form f (x) := h(??(x)), where h : Y ??? R is (d Y , d R )-Lipschitz, which entails that f is Lipschitz provided ?? is (which is almost always the case in practice).

In other words, we compute DISPLAYFORM1 where F is a class of (d Y , d R )-Lipschitz functions.

Optimising over this objective may now proceed as usual via weight-clipping like , or via a gradient penalty like BID8 .

Note that this suggestion may be understood as training a standard WGAN with the initial layers of the discriminator fixed to the embedding ??; our analysis here shows that this is equivalent to optimising with respect to W L 2 ????? instead of W L 2 .

We have begun some experimentation in this area but leave a more detailed empirical inquiry to future work.

It is also clearly important to establish better theoretical guarantees for our method.

At present, we have no guarantee that the number of samples in A and Y are enough to ensure that DISPLAYFORM2 (perhaps with some fixed bias that is fairly independent of ??, so that it is valid to use the value of D(??,?? ) to compare different choices of ??), or that (5) entails (2) with high probability.

We do however note that some recent theoretical work on the convergence rate of empirical Wasserstein estimations BID18 does suggest that it is plausible to hope for fast convergence of D(??,?? ) to D(??(X),?? ).

We also believe that the convincing empirical behaviour of W L 2 ????? does suggest that it is possible to say something more substantial about our approach, which we leave to future work.

The maximum mean discrepancy (MMD) is another well-known notion of distance on probability distributions, which has been used for testing whether two distributions are the same or not BID7 and also for learning generative models in the style of GAN BID11 BID5 BID15 BID10 .

It is parameterised by a characteristic kernel k, and defines the distance between probability distributions by means of the distance of k's reproducing kernel Hilbert space (RKHS).

The MMD induces the same weak topology on distributions as the one of the Wasserstein distance.

Under a mild condition, the MMD between distributions P and Q under a kernel k can be understood as the outcome of the following two-step calculation.

First, we pushforward P and Q from their original space X to a Hilbert space H (isomorphic to k's RKHS) using a feature function ?? : X ??? H induced by the kernel k. Typically, H is an infinite-dimensional space, such as the set 2 of square-summable sequences in R ??? as in Mercer's theorem.

Let P and Q be the resulting distributions on the feature space.

Second, we compute the supremum of E P (X) [f (X)] ??? E Q (X) [f (X)] over all linear 1-Lipschitz functions f : H ??? R. The result of this calculation is the MMD between P and Q.This two-step calculation shows the key difference between MMD and our use of Wasserstein distance and neural embedding ??.

While the co-domain of the feature function ?? is an infinitedimensional space (e.g. 2 ) in most cases, its counterpart ?? in our setting uses a finite-dimensional space as co-domain.

This means that the MMD possibly uses a richer feature space than our approach.

On the other hand, the MMD takes the supremum over only linear functions f among all 1-Lipschitz functions, whereas the Wasserstein distance considers all these 1-Lipschitz functions.

These different balancing acts between the expressiveness of features and that of functions taken in the supremum affect the learning and testing of various GAN approaches as observed experimentally in the literature.

One interesting future direction is to carry out systematic study on the theoretical and practical implications of these differences.

(

@highlight

Assess whether or not your GAN is actually doing something other than memorizing the training data.

@highlight

Aims to provide a quality measure/test for GANs and proposes to evaluate the current approximation of a distribution learnt by a GAN by using Wasserstein distance between two distributions made of a sum of Diracs as a baseline performance. 

@highlight

This paper proposed a procedure for assessing the performance of GANs by re-considering the key of observation, using the procedure to test and improve the current GANs