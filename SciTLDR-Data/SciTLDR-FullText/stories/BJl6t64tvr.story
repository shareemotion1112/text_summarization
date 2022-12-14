A commonplace belief in the machine learning community is that using adaptive gradient methods hurts generalization.

We re-examine this belief both theoretically and experimentally, in light of insights and trends from recent years.

We revisit some previous oft-cited experiments and theoretical accounts in more depth, and provide a new set of experiments in larger-scale, state-of-the-art settings.

We conclude that with proper tuning, the improved training performance of adaptive optimizers does not in general carry an overfitting penalty, especially in contemporary deep learning.

Finally, we synthesize a ``user's guide'' to adaptive optimizers, including some proposed modifications to AdaGrad to mitigate some of its empirical shortcomings.

Adaptive gradient methods have remained a cornerstone of optimization for deep learning.

They revolve around a simple idea: scale the step sizes according to the observed gradients along the execution.

It is generally believed that these methods enjoy accelerated optimization, and are more robust to hyperparameter choices.

For these reasons, adaptive optimizers have been applied across diverse architectures and domains.

However, in recent years, there has been renewed scrutiny on the distinction between adaptive methods and "vanilla" stochastic gradient descent (SGD).

Namely, several lines of work have purported that SGD, while often slower to converge, finds solutions that generalize better: for the same optimization error (training error), adaptive gradient methods will produce models with a higher statistical error (holdout validation error).

This claim, which can be shown to be true in a convex overparameterized examples, has perhaps muddled the consensus between academic research and practitioners pushing the empirical state of the art.

For the latter group, adaptive gradient methods have largely endured this criticism, and remain an invaluable instrument in the deep learning toolbox.

In this work, we revisit the generalization performance of adaptive gradient methods from an empirical perspective, and examine several often-overlooked factors which can have a significant effect on the optimization trajectory.

Addressing these factors, which does not require trying yet another new optimizer, can often account for what appear to be performance gaps between adaptive methods and SGD.

Our experiments suggest that adaptive gradient methods do not necessarily incur a generalization penalty: if an experiment indicates as such, there are a number of potential confounding factors and simple fixes.

We complete the paper with a discussion of inconsistent evidence for the generalization penalty of adaptive methods, from both experimental and theoretical viewpoints.

Our work investigates generalization of adaptive gradient methods, and constructively comments on the following:

The brittleness of simple experiments and simple abstractions.

We attempt a replication of the experiments from Wilson et al. (2017) , finding that they have not stood up to unknown hardware and software differences.

We show simple theoretical settings where adaptive methods can either fail or succeed dramatically, as compared to SGD.

Though each can shed interesting insights, neither abstraction is reflective of the truth.

The perils of choosing a large .

The innocuous initial accumulator value hyperparameter destroys adaptivity at parameter scales smaller than ??? .

This really matters in large-scale NLP; a foolproof solution is to use our proposed "?? = 0" variant of AdaGrad.

The subtleties in conducting a proper optimizer search.

The differences between Adam, AdaGrad, and RMSprop are not fundamental; some, like AdaGrad's lack of momentum, are easily fixable.

Upon disentangling these differences, and with enough tuning of the learning rate schedule, we suggest that all three are equally good candidates in optimizer search, and can match or beat SGD.

Adaptive regularization was introduced along the AdaGrad algorithm in parallel in (Duchi et al., 2011; McMahan & Streeter, 2010) .

A flurry of extensions, heuristics and modifications followed, most notably RMSprop (Tieleman & Hinton, 2012) and Adam (Kingma & Ba, 2014) .

Today, these papers have been cited tens of thousands of times, and the algorithms they propose appear in every deep learning framework.

For an in-depth survey of the theory of adaptive regularization and its roots in online learning, see (Hazan, 2016) .

Upon a quick perusal of recent literature, there is plenty of evidence that adaptive methods continue to be relevant in the state of the art.

Adam in particular remains a staple in recent developments in fields such as NLP (Devlin et al., 2018; Yang et al., 2019; Liu et al., 2019) , deep generative modeling (Karras et al., 2017; Brock et al., 2018; Kingma & Dhariwal, 2018) , and deep reinforcement learning (Haarnoja et al., 2018) .

Adaptive methods have seen adoption in extremely large-scale settings, necessitating modifications to reduce memory consumption (Shazeer & Stern, 2018; Anil et al., 2019; .

In recent years, there have been various works attempting to quantify the generalization properties of SGD.

These varied perspectives include general analyses based on stability and early stopping (Hardt et al., 2015) , a characterization of the implicit bias in special separable cases (Gunasekar et al., 2018a; b) , and a more fine-grained analysis for neural networks exploiting their specific structure (Allen-Zhu & Li, 2019; Arora et al., 2019) .

More recently, there has been a growing interest towards understanding the interpolation regime for overparameterized function fitting, where SGD is often the basic object of analysis (Belkin et al., 2019; Mei & Montanari, 2019) .

Finally, empirical questions on the generalization of adaptive gradient methods were brought to the forefront by Wilson et al. (2017) , who exhibit empirical and theoretical situations where adaptive methods generalize poorly.

Building on this premise, Keskar & Socher (2017) suggest switching from Adam and SGD during training.

Smith & Topin (2019) develop a doctrine of "superconvergence" which eschews adaptive methods.

Reddi et al. (2018) point out some pathological settings where Adam fails to converge, and amends the algorithm accordingly.

Schneider et al. (2019) note some sociological problems leading to misleading research on optimizer selection, providing a benchmarking suite for fairer hyperparameter searches, with mixed preliminary conclusions.

We begin by reviewing the stochastic optimization setting, and giving rigorous definitions of the adaptive gradient methods commonly used in practice.

We will focus on stochastic optimization tasks of the form

where the expectation is over a random variable z whose distribution D is initially unknown; in machine learning, z often represents a pair (x, y) of an example x and its corresponding label y, drawn from an unknown population.

A stochastic optimization algorithm is given a sample z 1 , . . .

, z T ??? D from the underlying distribution, and produces a pointw ??? R d whose population loss F (w) is as close as possible to that of the minimizer w = arg min w F (w).

Often, iterative (first-order) optimization methods maintain a sequence of iterates w 1 , . . .

, w T and, at each step t, use the stochastic gradient

to form the next iterate w t+1 .

The simplest stochastic optimization method is Stochastic Gradient Descent (SGD), whose update rule at step t takes the form

where ?? t > 0 is a step size (or learning rate) parameter, whose scale and time-varying behavior are typically determined via hyperparameter search.

Adaptive gradient methods comprise a general family of iterative optimization algorithms which attempt to automatically adapt to anisotropic gradient and parameter sizes.

Often, an adaptive method will incorporate a different (adaptively computed) step size for each entry of the gradient vector.

More specifically, the most common adaptive methods divide each parameter's gradient update by a second-moment-based estimate of the scale of its historical gradients.

A concise way to unify this family of adaptive methods is given by following update equation (starting from an arbitrary initializer w 0 ):

The above update expresses a broad family of methods including SGD, momentum (i.e., Polyak's heavy-ball method), AdaGrad, RMSprop, and Adam.

The particular instantiations of the parameters ?? k , ?? k are summarized below: Table 1 : Parameter settings for common optimization algorithms in the unified framework of Equation 1.

Here,

2 denotes the entrywise square of a vector or matrix.

We omit the parameters in the adaptive methods, see the discussion in Section 3.1.

In this section, we compile some lesser-known practices in the usage of adaptive methods, which we have found to help consistently across large-scale experiments.

We emphasize that this collection is restricted to simple ideas, which do not add extraneous hyperparameters or algorithmic alternatives.

The general AdaGrad update, as originally proposed by Duchi et al. (2011) , includes a parameter to allow for convenient inversions.

Formally, the update looks like:

The inclusion of in the the original proposal of the above updates seems to have been made for convenience of presenting the theoretical results.

However, in practice, this parameter often turns out to be a parameter that should be tuned depending on the problem instance.

The default value of this parameter in the standard implementations of the algorithm tend to be quite high; e.g., in Tensorflow (Abadi et al., 2016 ) it is 0.1 which can be quite high.

As large values would result in AdaGrad reducing to SGD with an implicit 1/ ??? learning rate, and losing out on all adaptive properties (RMSprop and Adam implementations also have an equivalent epsilon parameter).

The effect can be seen in Figure 4 which shows that along many coordinates the accumulated second moments of the gradient are very small, even in the middle of training.

At least one work remarks that the ability to choose a large ?? in a secondmoment-based adaptive method might be a feature rather than a shortcoming; the smooth interpolation with SGD may improve the stability of more exotic optimizers.

This does not appear to be the case for diagonal-matrix adaptive methods, in the NLP setting investigated in this paper.

Instead, we suggest removing this hyperparameter altogether and justify it in Section 4.2, and performing the AdaGrad update with the pseudoinverse instead of the full inverse.

Then, the update is given by the following:

where A ??? denotes the Moore-Penrose pseudoinverse of A and with the preconditioning matrices updated as before.

The above means that if there is a coordinate for which the gradient has been 0 thus far we make no movement in that coordinate.

This fix which can similarly be applied to the full matrix version of AdaGrad, does not affect the regret guarantees of AdaGrad.

We provide an analysis in the Appendix B, verifying as a sanity check that the standard AdaGrad regret bounds continue to hold when ?? is completely removed.

A key distinction between AdaGrad, RMSprop and Adam is as follows: AdaGrad does not include momentum, and there is a per-parameter learning rate which is inverse of the accumulated gradient squares for that parameter.

RMSprop as described in uses exponential moving averaging rather than straightforward accumulation that AdaGrad relies on, and Adam modifies RMSprop to add momentum for the gradients along with a bias-correction factor.

Note that implementation of RMSprop can vary based on the software library; e.g., TensorFlow (Abadi et al., 2016) includes modification to include momentum, while Keras API (Chollet et al. (2015) ) does not.

We note that it is straightforward to extend AdaGrad to incorporate heavy-ball momentum, where we start with??? 0 = 0 (and from a certain initialization w 0 ) and iteratively update:

The original definition of the Adam optimizer (Kingma & Ba, 2014) includes a bias correction term, in which the moment estimates are multiplied by the time-varying scalars (1 ??? ?? t 1 ) and (1 ??? ?? t 2 ).

As mentioned in the original paper, the bias correction can equivalently be written as an update to the learning rate.

In the notation of Table 1 :

As can be seen from Figure 2 , for the typical values of ?? 1 = 0.9 and ?? 2 = 0.999, the effective multiplier on the learning rate essentially resembles an external warmup applied on top of the learning rate.

The general heuristic of including a warmup phase at the beginning of training has gained significant popularity in state-of-the-art empirical works; see, for example, Goyal et al. (2017) Applying such a warm up externally on Adam results in now 3 hyper-parameters (?? 1 , ?? 2 and now the amount of warmup) conflating with each other, making hyper-parameter tuning difficult.

Instead we suggest to complete disable this bias correction altogether and use an explicit warmup schedule in place of it.

We use such a schedule in all of our experiments for SGD as well as adaptive optimizers as we find that it helps consistently across language modelling experiments.

One motivation for warmup during the initial stages of training is that for adaptive updates, the squared norm of the preconditioned gradient during the initial stage is quite large compared to the scale of the parameters.

For the initial steps the preconditioned gradient squared norm is proportional to the number of coordinates with non-zero gradients where as the squared norm of the parameters are proportional to the number of nodes.

Therefore adaptive methods are naturally forced to start with a smaller learning rate.

The warmup in such a case helps the learning rate to rise up while the norm of the gradients fall sharply as training proceeds.

Learning rate decay schedules are one of the hardest to tune albeit a crucial hyperparameter of an optimizer.

Stochastic gradient like algorithms, domain specific learning rate schedules have been derived over time with a lot of care and effort, examples include Resnet-50 on ImageNet-2012 where state of the art configuration of SGD+Momentum follows a stair-case learning rate schedule (while other type of schedules maybe possible).

Adaptive algorithms apriori come with a potential promise of not requiring massive tuning of these schedules as they come with an in-built schedule with the caveat that AdaGrad variants like Adam or RMSprop does not enjoy a data-dependent decay like AdaGrad due to the presence of a non-zero decay factor and requires an external learning rate decay schedule.

Even for experiments in Kingma & Ba (2014) which introduces Adam optimizer has experiments to include a 1/ ??? T decay schedule for convergence.

In our experiments, we found this implicit decay of AdaGrad to be sufficient for achieving superior performance on training a machine translation model, while an external decay rate was necessary for training the Resnet-50 on ImageNet-2012 to high accuracy.

We study the empirical performance of various optimization methods for training large state-of-theart deep models, focusing on two domains: natural language processing (machine translation) and image recognition.

We study the convergence of various optimization methods when training a Transformer model (Vaswani et al., 2017) for machine translation.

We used the larger Transformer-Big architecture (Chen et al., 2018) ; this architecture has 6 layers in the encoder and decoder, with 1024 model dimensions, 8192 hidden dimensions, and 16 attention heads.

It was trained on the WMT'14 English to French dataset (henceforth "en???fr") that contains 36.3M sentence pairs.

All experiments were carried out on 32 cores of a TPU-v3 Pod (Jouppi et al., 2017) and makes use of the Lingvo (Shen et al., 2019) sequence-to-sequence TensorFlow library.

We compared several optimization methods for training; the results are reported in Fig. 3 .

We see that a properly tuned AdaGrad (with ?? = 0 and added momentum) outperforms Adam, while SGD with momentum, plain AdaGrad and RMSprop perform much worse on this task.

These results illustrate that adaptivity and momentum are both extremely effective in training these models.

In Section 3.1, we proposed an "?? = 0" variant of AdaGrad.

Here we empirically motivate this modification, by investigating the effect of the parameter ?? on the performance of AdaGrad.

We train the Transformer model from above on the en???fr dataset using AdaGrad while varying the value of ??.

The results are given in Fig. 4 .

We see drastic improvement in convergence as we lower the value of ?? down to 10 ???7 (lower values do not improve convergence further and are thus omitted from the figure).

To see where these dramatic improvements come from, we also visualize in ??

the histogram of the square gradient values for the embedding layer of the model at step t = 116200, which indicates that a large fraction of the cumulative gradient entries have extremely small magnitudes.

The choice of ?? is thus important, and justify our prescription of removing the dependency all-together instead of tuning it as a separate hyper-parameter.

Next, we trained a ResNet-50 architecture (He et al., 2015) on the Imagenet-2012 (Deng et al., 2009) dataset.

The task is to classify images as belonging to one of the 1000 classes.

Our training setup consists of 512 cores of a TPU v3 Pod and makes use of a relatively large batch size of 16386.

As a baseline, we considered SGD with momentum with a highly-tuned staircase learning rate schedule, that achieves 75.3% test accuracy after 90 epochs.

We compared several optimization methods on this task as seen in Fig. 5 : the straightforward application of AdaGrad (with a fixed ?? and with heavy ball momentum) achieves only a paltry 63.94% test accuracy.

Noticing that AdaGrad implicit decay schedule does not decay sufficiently fast, an external decay rate was added starting at epoch 50 of the form (1 ??? current epoch -50 50 ) 2 .

This change was sufficient for AdaGrad to reach a test accuracy of 74.76%-a drastic >??? 10% jump.

As demonstrated, learning rate schedule is a highly important hyperparameter and requires tuning for each task.

E.g., the baseline SGD is highly tuned and follows an elaborate stair case learning rate to reach 75% test accuracy.

We attempted to reproduce the experiments from Wilson et al. (2017) , using the same codebases and identical hyperparameter settings.

Although we were able to replicate some of their findings on these smaller-scale experiments, others appear to be sensitive to hyperparameter tuning, and perhaps subtle changes in the deep learning software and hardware stack that have occurred during the two years since the publication of that paper.

In this section, we summarize these findings.

Image classification.

On the classic benchmark task of CIFAR-10 classification with a VGG network (Simonyan & Zisserman, 2014) , we were able to replicate the (Wilson et al., 2017) results perfectly, using the same codebase 2 .

We repeated the hyperparameter search reported in the paper, found the same optimal base learning rates for each optimizer, and found the same stratification in performance between non-adaptive methods, Adam & RMSprop, and AdaGrad.

Character-level language modeling.

Curiously, our replication of the language modeling experiment using the same popular repository 3 was successful in reproducing the optimal hyperparameter settings, but resulted in an opposite conclusion.

Here, SGD found the objective with the smallest training loss, but Adam exhibited the best generalization performance.

We believe that software version discrepancies (our setup: CUDA 10.1, cuDNN 7.5.1) may account for these small differences.

Generative parsing.

We turn to the Penn Treebank (Marcus et al., 1994) constituency parsing code 4 accompanying (Choe & Charniak, 2016) .

Using the same architectural and training protocol modifications as specified in (Wilson et al., 2017) , we were able to get the model to converge with each optimizer.

However, for two of the settings (SGD and RMSprop), the best reported learning rates exhibited non-convergence (the fainter curves in Figure 6 ).

Similarly as the above experiment, the ranking of optimizers' training and generalization performance differs from that seen in the original report.

Finally, Wilson et al. (2017) include a fourth set of experiments, generative parsing of Penn Treebank, using the code 5 accompanying (Cross & Huang, 2016) .

Unfortunately, this DyNet (Neubig et al., 2017) implementation, which was last updated in 2016, encountered a fatal memory leak when training with our DyNet 2.1 setup.

All relevant plots are given in Figure 6 , with color codes selected to match Figures 1 and 2 in Wilson et al. (2017) .

Together, these experiments are further evidence for a widespread reproducibility crisis in deep learning: despite the authors' exceptional transparency in disclosing their optimizer selection and evaluation protocol, these benchmarks have turned out to be brittle for unknown reasons.

Along the same lines as the random-seed-tuning experiments of Henderson et al. (2018) , this suggests that there are further technical complications to the problems of credible optimizer evaluation addressed by Schneider et al. (2019) , even on well-known supervised learning benchmarks.

Character-level language modeling with a 2-layer LSTM.

The original reported hyperparameters are the best and all optimizers converge to reasonable solutions, but contradictory conclusions about generalization arise.

Bottom: 3-layer LSTM for generative parsing.

Training does not converge with all reported learning rates; conclusions about generalization are unclear.

In this section we provide two simple examples of stochastic convex problems where it can be seen that when it comes to generalization both AdaGrad and SGD can be significantly better than the other depending on the instance.

Our purpose to provide both the examples is to stress our point that the issue of understanding the generalization performance of SGD vs. adaptive methods is more nuanced than what simple examples might suggest and hence such examples should be treated as qualitative indicators more for the purpose of providing intuition.

Indeed which algorithm will perform better on a given problem, depends on various properties of the precise instance.

We provide a brief intuitive review of the construction provided by Wilson et al. (2017) ; for a precise description, see Section 3.3 of that paper.

Consider a setting of overparameterized linear regression, where the true output (i.e. dependent variable) y ??? {??1} is the first coordinate of the feature vector (independent variable)

x. The next two coordinates of x are "dummy" coordinates set to 1; then, the coordinates are arranged in blocks which only appear once per sample, taking the value of y.

The key idea is that in this setting, the solution space that AdaGrad explores is always in the subspace of the sign vector of X y. As a result, AdaGrad treats the first three coordinates essentially indistinguishably putting equal mass on each.

It can then be seen that for any new example the AdaGrad solution does not extract the true label information from the first three coordinates and hence gets the prediction wrong, leading to high generalization error; the other distinguishing features belong to the new unique block which are set to 0 for the AdaGrad solution, as it has not seen them.

This example is motivated from the original AdaGrad paper (Duchi et al., 2011) , adapted to the overparameterized setting.

Consider a distribution Z supported over {0, 1} d with equal 1/d mass over vectors with exactly one 1 and 0 mass everywhere else.

Let the label distribution be always y = 1.

Consider sampling a dataset S of size c ?? d where c ??? 1 (corresponding to the overparameterized setting) and consider the hinge loss f t (x) = [1 ??? y t (z t x t )] + where (z t , y t ) denotes the t-th (example, label) pair.

Note that there is an optimal predictor given by the all-ones vector.

Running AdaGrad in such a setting, it can be seen that the first time a vector that has not appeared yet is sampled, AdaGrad quickly adapts by setting the coordinate corresponding to the vector to 1 and thereby making 0 error on the example.

Therefore after one epoch of AdaGrad (cd steps), the training error reduces to 0 and the average test error becomes roughly (1 ??? c).

On the other hand, for SGD (with an optimal 1/ ??? t decay scheme) after say cd/2 steps, the learning rate reduces to at most O(1/ ??? d) and therefore in the next cd/2 steps SGD reduces the error at most by a factor of O(1 ??? 1 ??? d), leading to a total test error of at least ??? (1 ??? c/2) after a total of cd steps.

This is significantly smaller than the error achieved by AdaGrad at this stage.

Further note that to get down to the same test error as that achieved by AdaGrad, it can be seen that SGD requires at least ???( ??? d) times more steps than AdaGrad.

@highlight

Adaptive gradient methods, when done right, do not incur a generalization penalty. 