Momentum based stochastic gradient methods such as heavy ball (HB) and Nesterov's accelerated gradient descent (NAG) method are widely used in practice for training deep networks and other supervised learning models, as they often provide significant improvements over stochastic gradient descent (SGD).

Rigorously speaking, fast gradient methods have provable improvements over gradient descent only for the deterministic case, where the gradients are exact.

In the stochastic case, the popular explanations for their wide applicability is that when these fast gradient methods are applied in the stochastic case, they partially mimic their exact gradient counterparts, resulting in some practical gain.

This work provides a counterpoint to this belief by proving that there exist simple problem instances where these methods cannot outperform SGD despite the best setting of its parameters.

These negative problem instances are, in an informal sense, generic; they do not look like carefully constructed pathological instances.

These results suggest (along with empirical evidence) that HB or NAG's practical performance gains are a by-product of minibatching.



Furthermore, this work provides a viable (and provable) alternative, which, on the same set of problem instances, significantly improves over HB, NAG, and SGD's performance.

This algorithm, referred to as Accelerated Stochastic Gradient Descent (ASGD), is a simple to implement stochastic algorithm, based on a relatively less popular variant of Nesterov's Acceleration.

Extensive empirical results in this paper show that ASGD has performance gains over HB, NAG, and SGD.

The code for implementing the ASGD Algorithm can be found at https://github.com/rahulkidambi/AccSGD.

First order optimization methods, which access a function (to be optimized) through its gradient or an unbiased approximation of its gradient, are the workhorses for modern large scale optimization problems, which include training the current state-of-the-art deep neural networks.

Gradient descent (Cauchy, 1847) is the simplest first order method that is used heavily in practice.

However, it is known that for the class of smooth convex functions as well as some simple non-smooth problems (Nesterov, 2012a) ), gradient descent is suboptimal BID1 and there exists a class of algorithms called fast gradient/momentum based methods which achieve optimal convergence guarantees.

The heavy ball method BID5 ) and Nesterov's accelerated gradient descent (Nesterov, 1983 ) are two of the most popular methods in this category.

On the other hand, training deep neural networks on large scale datasets have been possible through the use of Stochastic Gradient Descent (SGD) BID10 , which samples a random subset of training data to compute gradient estimates that are then used to optimize the objective function.

The advantages of SGD for large scale optimization and the related issues of tradeoffs between computational and statistical efficiency was highlighted in Bottou & Bousquet (2007) .The above mentioned theoretical advantages of fast gradient methods BID5 Nesterov, 1983) (albeit for smooth convex problems) coupled with cheap to compute stochastic gradient estimates led to the influential work of BID16 , which demonstrated the empirical advantages possessed by SGD when augmented with the momentum machinery.

This work has led to widespread adoption of momentum methods for training deep neural nets; so much so that, in the context of neural network training, gradient descent often refers to momentum methods.

But, there is a subtle difference between classical momentum methods and their implementation in practice -classical momentum methods work in the exact first order oracle model BID1 , i.e., they employ exact gradients (computed on the full training dataset), while in practice BID16 , they are implemented with stochastic gradients (estimated from a randomly sampled mini-batch of training data).

This leads to a natural question:"Are momentum methods optimal even in the stochastic first order oracle (SFO) model, where we access stochastic gradients computed on a small constant sized minibatches (or a batchsize of 1?)"Even disregarding the question of optimality of momentum methods in the SFO model, it is not even known if momentum methods (say, BID5 ; Nesterov (1983) ) provide any provable improvement over SGD in this model.

While these are open questions, a recent effort of Jain et al. (2017) showed that improving upon SGD (in the stochastic first order oracle) is rather subtle as there exists problem instances in SFO model where it is not possible to improve upon SGD, even information theoretically.

Jain et al. (2017) studied a variant of Nesterov's accelerated gradient updates BID2 for stochastic linear regression and show that their method improves upon SGD wherever it is information theoretically admissible.

Through out this paper, we refer to the algorithm of Jain et al. (2017) as Accelerated Stochastic Gradient Method (ASGD) while we refer to a stochastic version of the most widespread form of Nesterov's method (Nesterov, 1983) as NAG; HB denotes a stochastic version of the heavy ball method BID5 .

Critically, while Jain et al. (2017) shows that ASGD improves on SGD in any information-theoretically admissible regime, it is still not known whether HB and NAG can achieve a similar performance gain.

A key contribution of this work is to show that HB does not provide similar performance gains over SGD even when it is informationally-theoretically admissible.

That is, we provide a problem instance where it is indeed possible to improve upon SGD (and ASGD achieves this improvement), but HB cannot achieve any improvement over SGD.

We validate this claim empirically as well.

In fact, we provide empirical evidence to the claim that NAG also do not achieve any improvement over SGD for several problems where ASGD can still achieve better rates of convergence.

This raises a question about why HB and NAG provide better performance than SGD in practice BID16 , especially for training deep networks.

Our conclusion (that is well supported by our theoretical result) is that HB and NAG's improved performance is attributed to mini-batching and hence, these methods will often struggle to improve over SGD with small constant batch sizes.

This is in stark contrast to methods like ASGD, which is designed to improve over SGD across both small or large mini-batch sizes.

In fact, based on our experiments, we observe that on the task of training deep residual networks (He et al., 2016a) on the cifar-10 dataset, we note that ASGD offers noticeable improvements by achieving 5 ??? 7% better test error over HB and NAG even with commonly used batch sizes like 128 during the initial stages of the optimization.

The contributions of this paper are as follows.1.

In Section 3, we prove that HB is not optimal in the SFO model.

In particular, there exist linear regression problems for which the performance of HB (with any step size and momentum) is either the same or worse than that of SGD while ASGD improves upon both of them.2.

Experiments on several linear regression problems suggest that the suboptimality of HB in the SFO model is not restricted to special cases -it is rather widespread.

Empirically, the same holds true for NAG as well (Section 5).3.

The above observations suggest that the only reason for the superiority of momentum methods in practice is mini-batching, which reduces the variance in stochastic gradients and moves the SFO closer to the exact first order oracle.

This conclusion is supported by em-Algorithm 1 HB: Heavy ball with a SFO Require: Initial w 0 , stepsize ??, momentum ?? 1: w ???1 ??? w 0 ; t ??? 0 /*Set w ???1 to w 0 */ 2: while w t not converged do 3:w t+1 ??? w t ????? ?? ???f t (w t )+????(w t ??? w t???1 ) /*Sum of stochastic gradient step and momentum*/ DISPLAYFORM0 Note that for a given problem/input distribution?? = c is a constant while ?? = can be arbitrarily large.

Note that ?? >?? = c. Hence, ASGD improves upon rate of SGD by a factor of ??? ??.

The following proposition, which is the main result of this section, establishes that HB (Algorithm 1) cannot provide a similar improvement over SGD as what ASGD offers.

In fact, we show no matter the choice of parameters of HB, its performance does not improve over SGD by more than a constant.

Proposition 3.

Let w HB t be the t th iterate of HB (Algorithm 1) on the above problem with starting point w 0 .

For any choice of stepsize ?? and momentum ?? ??? [0, 1], ???T large enough such that ???t ??? T , we have, DISPLAYFORM1 where C(??, ??, ??) depends on ??, ?? and ?? (but not on t).Thus, to obtain w s.t.

w ??? w * ??? , HB requires ???(?? log 1 ) samples and iterations.

On the other hand, ASGD can obtain -approximation to w * in O( ??? ?? log ?? log 1 ) iterations.

We note that the gains offered by ASGD are meaningful when ?? > O(c) (Jain et al., 2017) ; otherwise, all the algorithms including SGD achieve nearly the same rates (upto constant factors).

While we do not prove it theoretically, we observe empirically that for the same problem instance, NAG also obtains nearly same rate as HB and SGD.

We conjecture that a lower bound for NAG can be established using a similar proof technique as that of HB (i.e. Proposition 3).

We also believe that the constant in the lower bound described in proposition 3 can be improved to some small number (??? 5).

We will now present and explain an intuitive version of ASGD (pseudo code in Algorithm 3).

The algorithm takes three inputs: short step ??, long step parameter ?? and statistical advantage parameter ??.

The short step ?? is precisely the same as the step size in SGD, HB or NAG.

For convex problems, this scales inversely with the smoothness of the function.

The long step parameter ?? is intended to give an estimate of the ratio of the largest and smallest curvatures of the function; for convex functions, this is just the condition number.

The statistical advantage parameter ?? captures trade off between statistical and computational condition numbers -in the deterministic case, ?? = ??? ?? and ASGD is equivalent to NAG, while in the high stochasticity regime, ?? is much smaller.

The algorithm maintains two iterates: descent iterate w t and a running averagew t .

The running average is a weighted average of the previous average and a long gradient step from the descent iterate, while the descent iterate is updated as a convex combination of short gradient step from the descent iterate and the running average.

The idea is that since the algorithm takes a long step as well as short step and an appropriate average of both of them, it can make progress on different directions at a similar pace.

Appendix B shows the equivalence between Algorithm 3 and ASGD as proposed in Jain et al. (2017) .

Note that the constant 0.7 appearing in Algorithm 3 has no special significance.

Jain et al. (2017) require it to be smaller than 1/6 but any constant smaller than 1 seems to work in practice.

We now present our experimental results exploring performance of SGD, HB, NAG and ASGD.

Our experiments are geared towards answering the following questions:??? Even for linear regression, is the suboptimality of HB restricted to specific distributions in

In this section, we will present results on performance of the four optimization methods (SGD, HB, NAG, and ASGD) for linear regression problems.

We consider two different class of linear regression problems, both of them in two dimensions.

Given ?? which stands for condition number, we consider the following two distributions:Discrete: a = e 1 w.p.

0.5 and a = 2 ?? ?? e 2 with 0.5; e i is the i th standard basis vector.

Gaussian : a ??? R 2 is distributed as a Gaussian random vector with covariance matrix 1 0 0 DISPLAYFORM0 We fix a randomly generated w * ??? R 2 and for both the distributions above, we let b = w * , a .

We vary ?? from {2 4 , 2 5 , ..., 2 12 } and for each ?? in this set, we run 100 independent runs of all four methods, each for a total of t = 5?? iterations.

We define that the algorithm converges if there is no error in the second half (i.e. after 2.5?? updates) that exceeds the starting error -this is reasonable since we expect geometric convergence of the initial error.

Unlike ASGD and SGD, we do not know optimal learning rate and momentum parameters for NAG and HB in the stochastic gradient model.

So, we perform a grid search over the values of the learning rate and momentum parameters.

In particular, we lay a 10 ?? 10 grid in [0, 1] ?? [0, 1] for learning rate and momentum and run NAG and HB.

Then, for each grid point, we consider the subset of 100 trials that converged and computed the final error using these.

Finally, the parameters that yield the minimal error are chosen for NAG and HB, and these numbers are reported.

We measure convergence performance of a method using: DISPLAYFORM1 Figure 1: Plot of 1/rate (refer equation FORMULA3 We compute the rate (1) for all the algorithms with varying condition number ??.

Given a rate vs ?? plot for a method, we compute it's slope (denoted as ??) using linear regression.

Table 1 presents the estimated slopes (i.e. ??) for various methods for both the discrete and the Gaussian case.

The slope values clearly show that the rate of SGD, HB and NAG have a nearly linear dependence on ?? while that of ASGD seems to scale linearly with ??? ??.

In this section, we present experimental results on training deep autoencoders for the mnist dataset, and we closely follow the setup of Hinton & Salakhutdinov (2006) .

This problem is a standard benchmark for evaluating the performance of different optimization algorithms e.g., Martens (2010); BID16 Martens & Grosse (2015) ; BID9 .

The network architecture follows previous work (Hinton & Salakhutdinov, 2006) and is represented as 784 ??? 1000 ??? 500 ??? 250???30???250???500???1000???784 with the first and last 784 nodes representing the input and output respectively.

All hidden/output nodes employ sigmoid activations except for the layer with 30 nodes which employs linear activations and we use MSE loss.

Initialization follows the scheme of Martens (2010), also employed in BID16 Martens & Grosse (2015) .

We perform training with two minibatch sizes ???1 and 8.

The runs with minibatch size of 1 were run for 30 epochs while the runs with minibatch size of 8 were run for 50 epochs.

For each of SGD, HB, NAG and ASGD, a grid search over learning rate, momentum and long step parameter (whichever is applicable) was done and best parameters were chosen based on achieving the smallest training error in the same protocol followed by BID16 .

The grid was extended whenever the best parameter fell at the edge of a grid.

For the parameters chosen by grid search, we perform 10 runs with different seeds and averaged the results.

The results are presented in Figures 2 and 3.

Note that the final loss values reported are suboptimal compared to those in published literature e.g., BID16 ; while BID16 report results after 750000 updates with a large batch size of 200 (which implies a total of 750000 ?? 200 = 150M gradient evaluations), whereas, our results are after 1.8M updates of SGD with a batch size 1 (which is just 1.8M gradient evaluations).Effect of minibatch sizes: While HB and NAG decay the loss faster compared to SGD for a minibatch size of 8 FIG1 ), this superior decay rate does not hold for a minibatch size of 1 FIG2 ).

This supports our intuitions from the stochastic linear regression setting, where we demonstrate that HB and NAG are suboptimal in the stochastic first order oracle model.

Comparison of ASGD with momentum methods: While ASGD performs slightly better than NAG for batch size 8 in the training error FIG1 ), ASGD decays the error at a faster rate compared to all the three other methods for a batch size of 1 FIG2 ).

We will now present experimental results on training deep residual networks (He et al., 2016b) with pre-activation blocks He et al. (2016a) for classifying images in cifar-10 (Krizhevsky & Hinton, 2009); the network we use has 44 layers (dubbed preresnet-44).

The code for this section was downloaded from preresnet (2017).

One of the most distinct characteristics of this experiment compared to our previous experiments is learning rate decay.

We use a validation set based decay scheme, wherein, after every 3 epochs, we decay the learning rate by a certain factor (which we grid search on) if the validation zero one error does not decrease by at least a certain amount (precise numbers are provided in the appendix since they vary across batch sizes).

Due to space constraints, we present only a subset of training error plots.

Please see Appendix C.3 for some more plots on training errors.

Effect of minibatch sizes: Our first experiment tries to understand how the performance of HB and NAG compare with that of SGD and how it varies with minibatch sizes.

FIG3 presents the test zero one error for minibatch sizes of 8 and 128.

While training with batch size 8 was done for 40 epochs, with batch size 128, it was done for 120 epochs.

We perform a grid search over all parameters for each of these algorithms.

See Appendix C.3 for details on the grid search parameters.

We observe that final error achieved by SGD, HB and NAG are all very close for both batch sizes.

While NAG exhibits a superior rate of convergence compared to SGD and HB for batch size 128, this superior rate of convergence disappears for a batch size of 8.

The next experiment tries to understand how ASGD compares with HB and NAG.

The errors achieved by various methods when we do grid search over all parameters are presented in While the final error achieved by ASGD is similar/favorable compared to all other methods, we are also interested in understanding whether ASGD has a superior convergence speed.

For this experiment, we need to address the issue of differing learning rates used by various algorithms and different iterations where they decay learning rates.

So, for each of HB and NAG, we choose the learning rate and decay factors by grid search, use these values for ASGD and do grid search only over long step parameter ?? and momentum ?? for ASGD.

The results are presented in Figures 5 and 6.

For batch size 128, ASGD decays error at a faster rate compared to both HB and NAG.

For batch size 8, while we see a superior convergence of ASGD compared to NAG, we do not see this superiority over HB.

The reason for this turns out to be that the learning rate for HB, which we also use for ASGD, turns out to be quite suboptimal for ASGD.

So, for batch size 8, we also compare fully optimized (i.e., grid search over learning rate as well) ASGD with HB.

The superiority of ASGD over HB is clear from this comparison.

These results suggest that ASGD decays error at a faster rate compared to HB and NAG across different batch sizes.

First order oracle methods: The primary method in this family is Gradient Descent (GD) (Cauchy, 1847).

As mentioned previously, GD is suboptimal for smooth convex optimization (Nesterov, Figure 6 : Test zero one loss for batch size 128 (left), batch size 8 (center) and training function value for batch size 8 (right) for ASGD compared to NAG.

In the above plots, ASGD was run with the learning rate and decay schedule of NAG.

Other parameters were selected by grid search.2004), and this is addressed using momentum methods such as the Heavy Ball method BID5 ) (for quadratics), and Nesterov's Accelerated gradient descent (Nesterov, 1983) .Stochastic first order methods and noise stability: The simplest method employing the SFO is SGD BID10 ; the effectiveness of SGD has been immense, and its applicability goes well beyond optimizing convex objectives.

Accelerating SGD is a tricky proposition given the instability of fast gradient methods in dealing with noise, as evidenced by several negative results which consider statistical BID7 Polyak, 1987; BID12 , numerical BID4 Greenbaum, 1989) While HB BID5 and NAG (Nesterov, 1983 ) are known to be effective in case of exact first order oracle, for the SFO, the theoretical performance of HB and NAG is not well understood.

Understanding Stochastic Heavy Ball: Understanding HB's performance with inexact gradients has been considered in efforts spanning several decades, in many communities like controls, optimization and signal processing.

Polyak (1987) considered HB with noisy gradients and concluded that the improvements offered by HB with inexact gradients vanish unless strong assumptions on the inexactness was considered; an instance of this is when the variance of inexactness decreased as the iterates approach the minimizer.

BID7 ; BID12 ; BID15 suggest that the improved non-asymptotic rates offered by stochastic HB arose at the cost of worse asymptotic behavior.

We resolve these unquantified improvements on rates as being just constant factors over SGD, in stark contrast to the gains offered by ASGD.

Loizou & Richt??rik (2017) state their method as Stochastic HB but require stochastic gradients that nearly behave as exact gradients; indeed, their rates match that of the standard HB method BID5 .

Such rates are not information theoretically possible (see Jain et al. FORMULA3 ), especially with a batch size of 1 or even with constant sized minibatches.

Accelerated and Fast Methods for finite-sums: There have been developments pertaining to faster methods for finite-sums (also known as offline stochastic optimization): amongst these are methods such as SDCA BID13 , SAG BID11 , SVRG (Johnson & Zhang, 2013) , SAGA (Defazio et al., 2014) , which offer linear convergence rates for strongly convex finite-sums, improving over SGD's sub-linear rates BID8 .

These methods have been improved using accelerated variants BID14 Frostig et al., 2015a; Lin et al., 2015; Defazio, 2016; BID0 .

Note that these methods require storing the entire training set in memory and taking multiple passes over the same for guaranteed progress.

Furthermore, these methods require computing a batch gradient or require memory requirements (typically ???(| training data points|)).

For deep learning problems, data augmentation is often deemed necessary for achieving good performance; this implies computing quantities such as batch gradient (or storage necessities) over this augmented dataset is often infeasible.

Such requirements are mitigated by the use of simple streaming methods such as SGD, ASGD, HB, NAG.

For other technical distinctions between the offline and online stochastic methods refer to Frostig et al. (2015b) .Practical methods for training deep networks: Momentum based methods employed with stochastic gradients BID16 have become standard and very popular in practice.

These schemes tend to outperform standard SGD on several important practical problems.

As previously mentioned, we attribute this improvement to effect of mini-batching rather than improvement offered by HB or NAG in the SFO model.

Schemes such as Adagrad (Duchi et al., 2011) , RMSProp BID17 , Adam (Kingma & Ba, 2014) represent an important and useful class of algorithms.

The advantages offered by these methods are orthogonal to the advantages offered by fast gradient methods; it is an important direction to explore augmenting these methods with ASGD as opposed to standard HB or NAG based acceleration schemes.

Chaudhari et al. FORMULA3 proposed Entropy-SGD, which is an altered objective that adds a local strong convexity term to the actual empirical risk objective, with an aim to improve generalization.

However, we do not understand convergence rates for convex problems or the generalization ability of this technique in a rigorous manner.

Chaudhari et al. (2017) propose to use SGD in their procedure but mention that they employ the HB/NAG method in their implementation for achieving better performance.

Naturally, we can use ASGD in this context.

Path normalized SGD BID3 is a variant of SGD that alters the metric on which the weights are optimized.

As noted in their paper, path normalized SGD could be improved using HB/NAG (or even the ASGD method).

In this paper, we show that the performance gain of HB over SGD in stochastic setting is attributed to mini-batching rather than the algorithm's ability to accelerate with stochastic gradients.

Concretely, we provide a formal proof that for several easy problem instances, HB does not outperform SGD despite large condition number of the problem; we observe this trend for NAG in our experiments.

In contrast, ASGD (Jain et al., 2017) provides significant improvement over SGD for these problem instances.

We observe similar trends when training a resnet on cifar-10 and an autoencoder on mnist.

This work motivates several directions such as understanding the behavior of ASGD on domains such as NLP, and developing automatic momentum tuning schemes BID18 .A SUBOPTIMALITY OF HB: PROOF OF PROPOSITION 3Before proceeding to the proof, we introduce some additional notation.

Let ?? DISPLAYFORM0 t+1 denote the concatenated and centered estimates in the j th direction for j = 1, 2.

DISPLAYFORM1 , j = 1, 2.Since the distribution over x is such that the coordinates are decoupled, we see that ?? (j) t+1 can be written in terms of ?? (j) t as: DISPLAYFORM2 t+1 denote the covariance matrix of ?? DISPLAYFORM3 with, B (j) defined as DISPLAYFORM4 We prove Proposition 3 by showing that for any choice of stepsize and momentum, either of the two holds:??? B (1) has an eigenvalue larger than 1, or,??? the largest eigenvalue of B (2) is greater than 1 ??? 500 ?? .

This is formalized in the following two lemmas.

Lemma 4.

If the stepsize ?? is such that ???? DISPLAYFORM5 (1) has an eigenvalue ??? 1.Lemma 5.

If the stepsize ?? is such that ???? DISPLAYFORM6 (2) has an eigenvalue of magnitude DISPLAYFORM7 Given this notation, we can now consider the j th dimension without the superscripts; when needed, they will be made clear in the exposition.

Denoting x def = ???? 2 and t def = 1 + ?? ??? x, we have: DISPLAYFORM8 The analysis goes via computation of the characteristic polynomial of B and evaluating it at different values to obtain bounds on its roots.

Lemma 6.

The characteristic polynomial of B is: DISPLAYFORM9 Proof.

We first begin by writing out the expression for the determinant: DISPLAYFORM10 expanding along the first column, we have: DISPLAYFORM11 Expanding the terms yields the expression in the lemma.

The next corollary follows by some simple arithmetic manipulations.

Corollary 7.

Substituting z = 1 ??? ?? in the characteristic equation of Lemma 6, we have: DISPLAYFORM12 Proof of Lemma 4.

The first observation necessary to prove the lemma is that the characteristic polynomial D(z) approaches ??? as z ??? ???, i.e., lim z?????? D(z) = +???.Next, we evaluate the characteristic polynomial at 1, i.e. compute D(1).

This follows in a straightforward manner from corollary (7) by substituting ?? = 0 in equation (2), and this yields, DISPLAYFORM13 As ?? < 1, x = ???? 2 > 0, we have the following by setting D(1) ??? 0 and solving for x: DISPLAYFORM14 Since D(1) ??? 0 and D(z) ??? 0 as z ??? ???, there exists a root of D(??) which is ??? 1.Remark 8.

The above characterization is striking in the sense that for any c > 1, increasing the momentum parameter ?? naturally requires the reduction in the step size ?? to permit the convergence of the algorithm, which is not observed when fast gradient methods are employed in deterministic optimization.

For instance, in the case of deterministic optimization, setting c = 1 yields ???? 2 1 < 2(1 + ??).

On the other hand, when employing the stochastic heavy ball method with x (j) = 2?? 2 j , we have the condition that c = 2, and this implies, ???? DISPLAYFORM15 We now prove Lemma 5.

We first consider the large momentum setting.

Lemma 9.

When the momentum parameter ?? is set such that 1 ??? 450/?? ??? ?? ??? 1, B has an eigenvalue of magnitude ??? 1 ??? 450 ?? .Proof.

This follows easily from the fact that det(B) DISPLAYFORM16 Remark 10.

Note that the above lemma holds for any value of the learning rate ??, and holds for every eigen direction of H. Thus, for "large" values of momentum, the behavior of stochastic heavy ball does degenerate to the behavior of stochastic gradient descent.

We now consider the setting where momentum is bounded away from 1.Corollary 11.

Consider B (2) , by substituting ?? = l/??, x = ???? min = c(???? 2 1 )/?? in equation (2) and accumulating terms in varying powers of 1/??, we obtain: DISPLAYFORM17 Substituting the value of l in equation (3) , the coefficient of DISPLAYFORM18 We will bound this term along with (3 DISPLAYFORM19 2 to obtain: DISPLAYFORM20 where, we use the fact that ?? < 1, l ??? 9.

The natural implication of this bound is that the terms that are lower order, such as O(1/?? 4 ) and O(1/?? 5 ) will be negative owing to the large constant above.

Let us verify that this is indeed the case by considering the terms having powers of O(1/?? 4 ) and O(1/?? 5 ) from equation (3) : DISPLAYFORM21 ?? 4 The expression above evaluates to ??? 0 given an upperbound on the value of c. The expression above follows from the fact that l ??? 9, ?? ??? 1.

3 ) and O(1/?? 2 ), in particular, DISPLAYFORM0 In both these cases, we used the fact that DISPLAYFORM1 ?? .

Finally, other remaining terms are negative.

Before rounding up the proof of the proposition, we need the following lemma to ensure that our lower bounds on the largest eigenvalue of B indeed affect the algorithm's rates and are true irrespective of where the algorithm is begun.

Note that this allows our result to be much stronger than typical optimization lowerbounds that rely on specific initializations to ensure a component along the largest eigendirection of the update operator, for which bounds are proven.

Lemma 13.

For any starting iterate w 0 = w * , the HB method produces a non-zero component along the largest eigen direction of B.Proof.

We note that in a similar manner as other proofs, it suffices to argue for each dimension of the problem separately.

But before we start looking at each dimension separately, let us consider the j th dimension, and detail the approach we use to prove the claim: the idea is to examine the subspace spanned by covariance E ?? DISPLAYFORM2 2 , ..., for every starting iterate ?? DISPLAYFORM3 in the largest eigen direction of B (j) , and this decays at a rate that is at best ?? max (B (j) ).Since B (j) ??? R 4??4 , we begin by examining the expected covariance spanned by the iterates DISPLAYFORM4 This implies that k just appears as a scale factor.

This in turn implies that in order to analyze the subspace spanned by the covariance of iterates ?? DISPLAYFORM5 1 , ..., we can assume k (j) = 1 without any loss in generality.

This implies, ?? DISPLAYFORM6 Note that with this in place, we see that we can now drop the superscript j that represents the dimension, since the analysis decouples across the dimensions j ??? {1, 2}. Furthermore, let the entries of the vector ?? k be represented as DISPLAYFORM7 Furthermore, DISPLAYFORM8 Let us consider the vectorized form of ?? j = E [?? j ??? ?? j ], and we denote this as vec(?? j ).

Note that vec(?? j ) makes ?? j become a column vector of size 4 ?? 1.

Now, consider vec(?? j ) for j = 0, 1, 2, 3 and concatenate these to form a matrix that we denote as D, i.e. DISPLAYFORM9 Now, since we note that ?? j is a symmetric 2 ?? 2 matrix, D should contain two identical rows implying that it has an eigenvalue that is zero and a corresponding eigenvector that is 0 ???1/ ??? 2 1/ (2) 0 .

It turns out that this is also an eigenvector of B with an eigenvalue ??.

Note that det(B) = ?? 4 .

This implies there are two cases that we need to consider: (i) when all eigenvalues of B have the same magnitude (= ??).

In this case, we are already done, because there exists at least one non zero eigenvalue of D and this should have some component along one of the eigenvectors of B and we know that all eigenvectors have eigenvalues with a magnitude equal to ?? max (B).

Thus, there exists an iterate which has a non-zero component along the largest eigendirection of B.(ii) the second case is the situation when we have eigenvalues with different magnitudes.

In this case, note that det(B) = ?? 4 < (?? max (B)) 4 implying ?? max (B) > ??.

In this case, we need to prove that D spans a three-dimensional subspace; if it does, it contains a component along the largest eigendirection of B which will round up the proof.

Since we need to understand whether D spans a three dimensional subspace, we can consider a different (yet related) matrix, which we call R and this is defined as: DISPLAYFORM10 If we compute and prove that det(R) = 0, we are done since that implies that R has three non-zero eigenvalues.

This implies, we first define the following: let q ?? = (t ??? ??) 2 + (c ??? 1)x 2 .

Then, R can be expressed as: DISPLAYFORM11 Then, DISPLAYFORM12 Note that this determinant can be zero when DISPLAYFORM13 We show this is not possible by splitting our argument into two parts, one about the convergent regime of the algorithm (where, ???? DISPLAYFORM14 .We will now prove that ?? DISPLAYFORM15 is much larger than one allowed by the convergence of the HB updates, i.e., ???? DISPLAYFORM16 .

In particular, if we prove that DISPLAYFORM17 for any admissible value of ??, we are done.

c+(c???2)?? is true.

DISPLAYFORM18 We need to prove that the determinant does not vanish in the divergent regime for rounding up the proof to the lemma.

Now, let us consider the divergent regime of the algorithm, i.e., when, ???? DISPLAYFORM19 c+(c???2)?? .

Furthermore, for the larger eigendirection, the determinant is zero when ???? for all admissible values of c, we are done.

We will explore this in greater detail: DISPLAYFORM20 considering the quadratic in the left hand size and solving it for c, we have: DISPLAYFORM21 This holds true iff DISPLAYFORM22 Which is true automatically since c > 2.

This completes the proof of the lemma.

We are now ready to prove Lemma 5.Proof of Lemma 5.

Combining Lemmas 9 and 12, we see that no matter what stepsize and momentum we choose, B (j) has an eigenvalue of magnitude at least 1 ??? 500 ?? for some j ??? {1, 2}. This proves the lemma.

We begin by writing out the updates of ASGD as written out in Jain et al. (2017) , which starts with two iterates a 0 and d 0 , and from time t = 0, 1, ...T ??? 1 implements the following updates: DISPLAYFORM0 DISPLAYFORM1 Next, we specify the step sizes ?? 1 = c 2 3 / ??? ?? ??, ?? 1 = c 3 /(c 3 + ??), ?? 1 = ??/(c 3 ?? min ) and ?? 1 = 1/R 2 , where ?? = R 2 /?? min .

Note that the step sizes in the paper of Jain et al. (2017) with c 1 in their paper set to 1 yields the step sizes above.

Now, substituting equation 8 in equation 9 and substituting the value of ?? 1 , we have: DISPLAYFORM2 We see that d t+1 is precisely the update of the running averagew t+1 in the ASGD method employed in this paper.

We now update b t to become b t+1 and this can be done by writing out equation 6 at t + 1, i.e: DISPLAYFORM3 By substituting the value of ?? 1 we note that this is indeed the update of the iterate as a convex combination of the current running average and a short gradient step as written in this paper.

In this paper, we set c 3 to be equal to 0.7, and any constant less than 1 works.

In terms of variables, we note that ?? in this paper's algorithm description maps to 1 ??? ?? 1 .

In this section, we will present more details on our experimental setup.

In this section, we will present some more results on our experiments on the linear regression problem.

Just as in Appendix A, it is indeed possible to compute the expected error of all the algorithms among SGD, HB, NAG and ASGD, by tracking certain covariance matrices which evolve as linear systems.

For SGD, for instance, denoting ?? DISPLAYFORM0 , where B is a linear operator acting on d ?? d matrices such that DISPLAYFORM1 Similarly, HB, NAG and ASGD also have corresponding operators (see Appendix A for more details on the operator corresponding to HB).

The largest magnitude of the eigenvalues of these matrices indicate the rate of decay achieved by the particular algorithm -smaller it is compared to 1, faster the decay.

We now detail the range of parameters explored for these results: the condition number ?? was varied from {2 4 , 2 5 , .., 2 28 } for all the optimization methods and for both the discrete and gaussian problem.

For each of these experiments, we draw 1000 samples and compute the empirical estimate of the fourth moment tensor.

For NAG and HB, we did a very fine grid search by sampling 50 values in the interval (0, 1] for both the learning rate and the momentum parameter and chose the parameter setting that yielded the smallest ?? max (B) that is less than 1 (so that it falls in the range of convergence of the algorithm).

As for SGD and ASGD, we employed a learning rate of 1/3 for the Gaussian case and a step size of 0.9 for the discrete case.

The statistical advantage parameter of ASGD was chosen to be 3??/2 for the Gaussian case and 2??/3 for the Discrete case, and the a long step parameters of 3?? and 2?? were chosen for the Gaussian and Discrete case respectively.

The reason it appears as if we choose a parameter above the theoretically maximal allowed value of the advantage parameter is because the definition of ?? is different in this case.

The ?? we speak about for this experiment is ?? max /?? min unlike the condition number for the stochastic optimization problem.

In a manner similar to actually running the algorithms (the results of whose are presented in the main paper), we also note that we can compute the rate as in equation 1 and join all these rates using a curve and estimate its slope (in the log scale).

This result is indicated in table 3.

Figure 7 presents these results, where for each method, we did grid search over all parameters and chose parameters that give smallest ?? max .

We see the same pattern as in Figure 1 from actual runs -SGD,HB and NAG all have linear dependence on condition number ??, while ASGD has a dependence of ??? ??.

Table 3 : Slopes (i.e. ??) obtained by fitting a line to the curves in Figure 7 .

A value of ?? indicates that the error decays at a rate of exp ???t ?? ?? .

A smaller value of ?? indicates a faster rate of error decay.

We begin by noting that the learning rates tend to vary as we vary batch sizes, which is something that is known in theory (Jain et al., 2016) .

Furthermore, we extend the grid especially whenever our best parameters of a baseline method tends to land at the edge of a grid.

The parameter ranges explored by our grid search are:Batch Size 1: (parameters chosen by running for 20 epochs)??? SGD: learning rate: {0.01, 0.01 ??? 10, 0.1, 0.1 ??? 10, 1, ??? 10, 5, 10, 20, 10 ??? 10, 40, 60, 80, 100.??? NAG/HB: learning rate: {0.01 ??? 10, 0.1, 0.1 ??? 10, 1, ??? 10, 10}, momentum {0, 0.5, 0.75, 0.9, 0.95, 0.97}.??? ASGD: learning rate: {2.5, 5}, long step {100.0, 1000.0}, advantage parameter {2.5, 5.0, 10.0, 20.0}.Batch Size 8: (parameters chosen by running for 50 epochs)??? SGD: learning rate: {0.001, 0.001 ??? 10.0, 0.01, 0.01 ??? 10, 0.1, 0.1 ??? 10, 1, ??? 10, 5, 10 , 10 ??? 10, 40, 60, 80, 100, 120, 140}.??? NAG/HB: learning rate: {5.0, 10.0, 20.0, 10 ??? 10, 40, 60}, momentum {0, 0.25, 0.5, 0.75, 0.9, 0.95}.??? ASGD: learning rate {40, 60}. For a long step of 100, advantage parameters of {1.5, 2, 2.5, 5, 10, 20}. For a long step of 1000, we swept over advantage parameters of {2.5, 5, 10}.

In this section, we will provide more details on our experiments on cifar-10, as well as present some additional results.

We used a weight decay of 0.0005 in all our experiments.

The grid search parameters we used for various algorithms are as follows.

Note that the ranges in which parameters such as learning rate need to be searched differ based on batch size (Jain et al., 2016) .

Furthermore, we tend to extrapolate the grid search whenever a parameter (except for the learning rate decay factor) at the edge of the grid has been chosen; this is done so that we always tend to lie in the interior of the grid that we have searched on.

Note that for the purposes of the grid search, we choose a hold out set from the training data and add it in to the training data after the parameters are chosen, for the final run.

Batch Size 8: Note: (i) parameters chosen by running for 40 epochs and picking the grid search parameter that yields the smallest validation 0/1 error.(ii) The validation set decay scheme that we use is that if the validation error does not decay by at least 1% every three passes over the data, we cut the learning rate by a constant factor (which is grid searched as described below).

The minimal learning rate to use is fixed to be 6.25 ?? 10 ???5 , so that we do not decay far too many times and curtail progress prematurely.??? SGD: learning rate: {0.0033, 0.01, 0.033, 0.1, 0.33}, learning rate decay factor {5, 10}.??? NAG/HB: learning rate: {0.001, 0.0033, 0.01, 0.033}, momentum {0.8, 0.9, 0.95, 0.97}, learning rate decay factor {5, 10}.??? ASGD: learning rate {0.01, 0.0330, 0.1}, long step {1000, 10000, 50000}, advantage parameter {5, 10}, learning rate decay factor {5, 10}.Batch Size 128: Note: (i) parameters chosen by running for 120 epochs and picking the grid search parameter that yields the smallest validation 0/1 error.(ii) The validation set decay scheme that we use is that if the validation error does not decay by at least 0.2% every four passes over the data, we cut the learning rate by a constant factor (which is grid searched as described below).

The minimal learning rate to use is fixed to be 1 ?? 10 ???3 , so that we do not decay far too many times and curtail progress prematurely.??? SGD: learning rate: {0.01, 0.03, 0.09, 0.27, 0.81}, learning rate decay factor {2, ??? 10, 5}.??? NAG/HB: learning rate: {0.01, 0.03, 0.09, 0.27}, momentum {0.5, 0.8, 0.9, 0.95, 0.97}, learning rate decay factor {2, ??? 10, 5}.??? ASGD: learning rate {0.01, 0.03, 0.09, 0.27}, long step {100, 1000, 10000}, advantage parameter {5, 10, 20}, learning rate decay factor {2, ??? 10, 5}.As a final remark, for any comparison across algorithms, such as, (i) ASGD vs. NAG, (ii) ASGD vs HB, we fix the starting learning rate, learning rate decay factor and decay schedule chosen by the best grid search run of NAG/HB respectively and perform a grid search over the long step and advantage parameter of ASGD.

In a similar manner, when we compare (iii) SGD vs NAG or, (iv) SGD vs. HB, we choose the learning rate, learning rate decay factor and decay schedule of SGD and simply sweep over the momentum parameter of NAG or HB and choose the momentum that offers the best validation error.

We now present plots of training function value for different algorithms and batch sizes.

Effect of minibatch sizes: FIG10 plots training function value for batch sizes of 128 and 8 for SGD, HB and NAG.

We notice that in the initial stages of training, NAG obtains substantial improvements compared to SGD and HB for batch size 128 but not for batch size 8.

Towards the end of training however, NAG starts decreasing the training function value rapidly for both the batch sizes.

The reason for this phenomenon is not clear.

Note however, that at this point, the test error has already stabilized and the algorithms are just overfitting to the data.

Comparison of ASGD with momentum methods: We now present the training error plots for ASGD compared to HB and NAG in Figures 9 and 10 respectively.

As mentioned earlier, in order to see a clear trend, we constrain the learning rate and decay schedule of ASGD to be the same as that of HB and NAG respectively, which themselves were learned using grid search.

We see

<|TLDR|>

@highlight

Existing momentum/acceleration schemes such as heavy ball method and Nesterov's acceleration employed with stochastic gradients do not improve over vanilla stochastic gradient descent, especially when employed with small batch sizes.