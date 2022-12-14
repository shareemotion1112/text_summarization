Stochastic Gradient Descent (SGD) with Nesterov's momentum is a widely used optimizer in deep learning, which is observed to have excellent generalization performance.

However, due to the large stochasticity, SGD with Nesterov's momentum is not robust, i.e., its performance may deviate significantly from the expectation.

In this work, we propose Amortized Nesterov's Momentum, a special variant of Nesterov's momentum which has more robust iterates, faster convergence in the early stage and higher efficiency.

Our experimental results show that this new momentum achieves similar (sometimes better) generalization performance with little-to-no tuning.

In the convex case, we provide optimal convergence rates for our new methods and discuss how the theorems explain the empirical results.

In recent years, Gradient Descent (GD) (Cauchy, 1847) and its variants have been widely used to solve large scale machine learning problems.

Among them, Stochastic Gradient Descent (SGD) (Robbins & Monro, 1951) , which replaces gradient with an unbiased stochastic gradient estimator, is a popular choice of optimizer especially for neural network training which requires lower precision.

Sutskever et al. (2013) found that using SGD with Nesterov's momentum (Nesterov, 1983; 2013b) , which was originally designed to accelerate deterministic convex optimization, achieves substantial speedups for training neural networks.

This finding essentially turns SGD with Nesterov's momentum into the benchmarking method of neural network design, especially for classification tasks (He et al., 2016b; a; Zagoruyko & Komodakis, 2016; Huang et al., 2017) .

It is observed that in these tasks, the momentum technique plays a key role in achieving good generalization performance.

Adaptive methods (Duchi et al., 2011; Kingma & Ba, 2015; Tieleman & Hinton, 2012; Reddi et al., 2018) , which are also becoming increasingly popular in the deep learning community, diagonally scale the gradient to speed up training.

However, Wilson et al. (2017) show that these methods always generalize poorly compared with SGD with momentum (both classical momentum (Polyak, 1964 ) and Nesterov's momentum).

In this work, we introduce Amortized Nesterov's Momentum, which is a special variant of Nesterov's momentum.

From users' perspective, the new momentum has only one additional integer hyper-parameter m to choose, which we call the amortization length.

Learning rate and momentum parameter of this variant are strictly aligned with Nesterov's momentum and by choosing m = 1, it recovers Nesterov's momentum.

This paper conducts an extensive study based on both empirical evaluation and convex analysis to identify the benefits of the new variant (or from users' angle, to set m apart from 1).

We list the advantages of Amortized Nesterov's Momentum as follows:

??? Increasing m improves robustness 1 .

This is an interesting property since the new momentum not only provides acceleration, but also enhances the robustness.

We provide an understanding of this property by analyzing the relation between convergence rate and m in the convex setting.

??? Increasing m reduces (amortized) iteration complexity.

??? A suitably chosen m boosts the convergence rate in the early stage of training and produces comparable final generalization performance.

???

It is easy to tune m. The performances of the methods are stable for a wide range of m and we prove that the methods converge for any valid choice of m in the convex setting.

??? If m is not too large, the methods obtain the optimal convergence rate in general convex setting, just like Nesterov's method.

The new variant does have some minor drawbacks: it requires one more memory buffer, which is acceptable in most cases, and it shows some undesired behaviors when working with learning rate schedulers, which can be addressed by a small modification.

Considering these pros and cons, we believe that the proposed variant can benefit many large-scale deep learning tasks.

Our high level idea is simple: the stochastic Nesterov's momentum can be unreliable since it is provided only by the previous stochastic iterate.

The iterate potentially has large variance, which may lead to a false momentum that perturbs the training process.

We thus propose to use the stochastic Nesterov's momentum based on several past iterates, which provides robust acceleration.

In other words, instead of immediately using an iterate to provide momentum, we put the iterate into an "amortization plan" and use it later.

We start with a review of SGD and Nesterov's momentum.

We discuss some subtleties in the implementation and evaluation, which contributes to the interpretation of our methods.

Notations In this paper, we use x ??? R d to denote the vector of model parameters.

?? and ??, ?? denote the standard Euclidean norm and inner product, respectively.

Scalar multiplication for v ??? R d and ?? ??? R is denoted as ?? ?? v. f : R d ??? R denotes the loss function to be minimized and ???f (x) represents the gradient of f evaluated at x. We denote the unbiased stochastic gradient estimator of ???f (x) as ???f i (x) with the random variable i independent of x (e.g., using mini-batch).

We use x 0 ??? R d to denote the initial guess.

SGD SGD has the following simple iterative scheme, where ?? ??? R denotes the learning rate:

Nesterov's momentum The original Nesterov's accelerated gradient (with constant step) (Nesterov, 1983; 2013b) has the following scheme 2 (y ??? R d , ??, ?? ??? R and y 0 = x 0 ):

where we call ?? ?? (y k+1 ??? y k ) the momentum.

By simply replacing ???f (x k ) with ???f i k (x k ), we obtain the SGD with Nesterov's momentum, which is widely used in deep learning.

To make this point clear, recall that the reformulation in Sutskever et al. (2013) (scheme (2) , also the Tensorflow (Abadi et al., 2016)

Here the notations are modified based on their equivalence to scheme (1).

It can be verified that schemes (2) and (3) are equivalent to (1) through v k = ?? ???1 ??(x k ???y k ) and v pt k = ?? ???1 ?? ???1 ??(y k ???x k ), respectively (see Defazio (2018) for other equivalent forms of scheme (1)).

Interestingly, both PyTorch and Tensorflow 3 track the values {x k }, which we refer to as M-SGD.

This choice allows a consistent implementation when wrapped in a generic optimization layer (Defazio, 2018) .

However, the accelerated convergence rate (in the convex case) is built upon {y k } (Nesterov, 2013b) and {x k } may not possess such a theoretical improvement.

We use OM-SGD to refer to the Original M-SGD that outputs {y k }.

SGD and M-SGD In order to study the features of momentum, in this work, we regard momentum as an add-on to plain SGD, which corresponds to fixing the learning rates 4 ?? = ??.

From the interpretation in Allen-Zhu & Orecchia (2017), ?? represents the learning rate for the gradient descent "inside" Nesterov's method.

To introduce the evaluation metrics of this paper, we report the results of training ResNet34 (He et al., 2016b) on CIFAR-10 (Krizhevsky et al., 2009) (our basic case study) using SGD and M-SGD in Figure 1 .

In this paper, all the multiple runs start with the same initial guess x 0 .

Figure 1a shows that Nesterov's momentum hurts the convergence in the first 60 epochs but accelerates the final convergence, which verifies the importance of momentum for achieving high accuracy.

Figure 1b depicts the robustness of M-SGD and SGD, which suggests that adding Nesterov's momentum slightly increases the uncertainty in the training process of SGD.

Train-batch loss vs. Full-batch loss In Figure 1c , train-batch loss stands for the average of batch losses forwarded in an epoch, which is commonly used to indicate the training process in deep learning.

Full-batch loss is the average loss over the entire training dataset evaluated at the end of each epoch.

In terms of optimizer evaluation, full-batch loss is much more informative than trainbatch loss as it reveals the robustness of an optimizer.

However, full-batch loss is too expensive to evaluate and thus we only measure it on small datasets.

On the other hand, test accuracy couples optimization and generalization, but since it is also evaluated at the end of the epoch, its convergence is similar to full-batch loss.

Considering the basic usage of momentum in deep learning, we mainly use test accuracy to evaluate optimizers.

We provide more discussion on this issue in Appendix C.2.

M-SGD vs. OM-SGD We also include OM-SGD in Figure 1a .

In comparison, the final accuracies of M-SGD and OM-SGD are 94.606% ?? 0.152% and 94.728% ?? 0.111% with average deviations at 1.040% and 0.634%, respectively.

This difference can be explained following the interpretation in Hinton (2012) that {x k } are the points after "jump" and {y k } are the points after "correction".

In this section, we formally introduce SGD with Amortized Nesterov's Momentum (AM1-SGD) in Algorithm 1 with the following remarks:

Options It can be verified that if m = 1, AM1-SGD with Option I degenerates to M-SGD and Option II corresponds to OM-SGD.

Just like the case for M-SGD and OM-SGD, the accelerated convergence rate is built upon Option II while Option I is easier to be implemented in a generic optimization layer 5 .

Intuitively, Option I is SGD with amortized momentum and Option II applies an m-iterations tail averaging on Option I. 4 Ma & Yarats (2019) observed that when effective learning rates ?? = ??(1 ??? ??) ???1 are fixed, M-SGD and SGD have similar performance.

We provide a discussion on this observation in Appendix C.1.

5 To implement Option II, we can either maintain another identical network for the shifted pointx or temporarily change the network parameters in the evaluation phase.

Input: Initial guess x 0 , learning rate ??, momentum ??, amortization length m, iteration number K.

if (k + 1) mod m = 0 then 5:

.

{adding amortized momentum} 6:x ???x + ,x + ??? 0.

end if 8: end for Output:

Option I: x, Option II:x.

* The symbol '???' denotes assignment.

We can improve the efficiency of Algorithm 1 by maintaining a running scaled momentum???

instead of the running averagex + , by replacing the following steps in Algorithm 1:

Step 5:

Step 6:x ???x + (1/m) ?????

Then, in one m-iterations loop, for each of the first m ??? 1 iterations, AM1-SGD requires 1 vector addition and 1 scaled vector addition.

At the m-th iteration, it requires 1 vector addition, 1 scalarvector multiplication and 3 scaled vector additions.

In comparison, M-SGD (standard PyTorch) requires 1 vector addition, 1 (in-place) scalar-vector multiplication and 2 scaled vector additions per iteration.

Thus, as long as m > 2, AM1-SGD has lower amortized cost than M-SGD.

For memory complexity, AM1-SGD requires one more auxiliary buffer than M-SGD.

Tuning m We did a parameter sweep for m in our basic case study.

We plot the final and the average deviation of test accuracies over 5 runs against m in Figure 2a .

Note that m = 1 corresponds to the results of M-SGD and OM-SGD, which are already given in Figure 1 .

From this empirical result, m introduces a trade-off between final accuracy and robustness (the convergence behaviors can be found in Appendix A.1).

Figure 2a suggests that m = 5 is a good choice for this task.

For simplicity, and also as a recommended setting, we fix m = 5 for the rest of experiments in this paper.

A momentum that increases robustness To provide a stronger justification, we ran 20 seeds with m = 5 in Figure 2b and the detailed data are given in Figure 3 & Table 1 .

The results show that the amortized momentum significantly increases the robustness.

Intuitively, the gap between Option I and Option II can be understood as the effect of tail averaging.

However, the large gap between Option I and SGD is somewhat mysterious: what Option I does is to inject a very large momentum 6 into SGD every m iterations.

It turns out that this momentum not only provides acceleration, but also helps the algorithm become more robust than SGD.

This observation basically differentiates AM1-SGD from a simple interpolation in-between M-SGD and SGD.

6 Amortized momentum ????(x + ???x) is expected to be much large than Nesterov's momentum ????(y k+1 ???y k ).

Table 1 : Detailed data of the curves in Figure 2b .

Best viewed in color.

We observed that when we use schedulers with a large decay factor and the momentum ?? is too large for the task (e.g., 0.995 for the task of this section), there would be a performance drop after the learning rate reduction.

We believe that it is caused by the different cardinalities of iterates being averaged inx + , which leads to a false momentum.

This issue is resolved by restarting the algorithm after each learning rate reduction inspired by (O'donoghue & Candes, 2015) .

We include more discussion and evidence in Appendix A.4.

Algorithm 2 AM2-SGD

k=0 is a sequence of uniformly random indexes.

If Option II is used,?? 0 = x 0 . {a running average for the point table ??} 1: for k = 0, . . . , K ??? 1 do 2:

3:

While enjoying an improved efficiency, AM1-SGD does not have identical iterations 7 , which to some extent limits its extensibility to other settings (e.g., asynchronous setting).

In this section, we propose a variant of Amortized Nesterov's Momentum (AM2-SGD, Algorithm 2) to address this problem.

To show the characteristics of AM2-SGD, we make the following remarks:

Trading memory for extensibility In expectation, the point table ?? stores the most recent m iterations and thus the output?? K is an m-iterations tail average, which connects to AM1-SGD.

The relation between AM1-SGD and AM2-SGD resembles that of SVRG (Johnson & Zhang, 2013) and SAGA (Defazio et al., 2014) , the most popular methods in finite-sum convex optimization: to reuse the information from several past iterates, we can either maintain a "snapshot" that aggregates the information or keep the iterates in a table.

A side-by-side comparison is given in Section 4.

Options and convergence As in the case of AM1-SGD, if m = 1, AM2-SGD with Option I corresponds to M-SGD and Option II is OM-SGD.

In our preliminary experiments, the convergence of AM2-SGD is similar to AM1-SGD and it also has the learning rate scheduler issue.

In our preliminary experiments (can be found in Appendix A), we observed that Option I is consistently worse than Option II and it does not seem to benefit from increasing m. Thus, we do not recommend using Option I. We also set m = 5 for AM2-SGD for its evaluation due to the similarity.

Additional randomness {j k } In our implementation, at each iteration, we sample an index in [m] as j k+1 and obtain the stored index j k .

We observed that with Option I, AM2-SGD has much larger deviations than AM1-SGD, which we believe is caused by the additional random indexes {j k }.

The original Nesterov's accelerated gradient is famous for its optimal convergence rates for solving convex problems.

In this section, we analyze the convergence rates for AM1-SGD and AM2-SGD in the convex case, which explicitly model the effect of amortization (i.e., m).

While these rates do not hold for deep learning problems in general, they help us understand the observed convergence behaviors of the proposed methods, especially on how they differ from M-SGD (m = 1).

Moreover, the analysis also provides intuition on tuning m. Since the original Nesterov's method is deterministic (Nesterov, 1983; 2013b) , we follow the setting of its stochastic variants (Lan, 2012; Ghadimi & Lan, 2012) , in which Nesterov's acceleration also achieves the optimal rates.

We consider the following convex composite problem (Beck & Teboulle, 2009; Nesterov, 2013a) :

where X ??? R d is a non-empty closed convex set and h is a proper convex function with its proximal operator prox ??h (??) 8 available.

We impose the following assumptions on the regularity of f and the stochastic oracle ???f i (identical to the ones in Ghadimi & Lan (2012) with ?? = 0):

The notation

. .

, i k???1 )] for a random process i 0 , i 1 , . .

..

These assumptions cover several important classes of convex problems.

For example, (a) covers the cases of f being L-smooth (M = 0) or L 0 -Lipschitz continuous (M = 2L 0 , L = 0) convex functions and if ?? = 0 in (c), the assumptions cover several classes of deterministic convex programming problems.

We denote x ??? X as a solution to problem (4) and x 0 ??? X as the initial guess.

Unlike its usage in deep learning, the momentum parameter ?? is always a variable in general convex analysis.

For the simplicity of analysis, we reformulate AM1-SGD (Algorithm 1) and AM2-SGD (Algorithm 2) into the following schemes 10 (z ??? X, ?? ??? R):

Initialize:

for j = 0, . . .

, m ??? 1 do 3:

4:

5:

:

end for 8:x s+1 = 1 m m j=1 x sm+j .

9: end for Output:x S .

Initialize:

3:

We show in Appendix B.1 that when h ??? 0 and ?? is a constant, the reformulated schemes AM1-SGD and AM2-SGD are equivalent to Algorithm 1 and Algorithm 2 through ?? s = ??(1 ??? ?? s ) ???1 and Parikh et al. (2014) .

9 When M > 0, f is not necessarily differentiable and we keep using the notation ???f (x) to denote an arbitrary subgradient of f at x for consistency.

10 For simplicity, we assume K is divisible by m.

.

These reformulations are basically how Nesterov's momentum was migrated into deep learning (Sutskever et al., 2013) .

Then we establish the convergence rates for AM1-SGD and AM2-SGD as follows.

All the proofs in this paper are given in Appendix B.2.

Theorem 1.

For the reformulated AM1-SGD, suppose we choose

Then,

(b) If the variance has a "light tail", i.e., E i exp ???f i (x)??????f (x) 2 /?? 2 ??? exp{1}, ???x ??? X, and X is compact, denoting D X max x???X x ??? x , for any ?? ??? 0, we have

Remarks: (a) Regarding K 0 (m), its minimum is obtained at either m = 1 or m = K. Note that for AM1-SGD, m is strictly constrained in {1, . . .

, K}. It can be verified that when m = K, AM1-SGD becomes the modified mirror descent SA (Lan, 2012), or under the Euclidean setting, the SGD that outputs the average of the whole history, which is rarely used in practice.

In this case, the convergence rate in Theorem 1a becomes the corresponding Understandings: Theorem 1a gives the expected performance in terms of full-batch loss F (x) ??? F (x ), from which the trade-off of m is clear: Increasing m improves the dependence on variance ?? but deteriorates the O(L/K 2 ) term (i.e., the acceleration).

Based on this trade-off, we can understand the empirical results in Figure 2b : the faster convergence in the early stage could be the result of a better control on ?? and the slightly lowered final accuracy is possibly caused by the reduced acceleration effect.

Theorem 1b provides the probability of the full-batch loss deviating from its expected performance (i.e., K 0 (m)).

It is clear that increasing m leads to smaller deviations with the same probability, which sheds light on the understanding of the increased robustness observed in Figure 2 .

Since the theorem is built on the full-batch loss, we did an experiments based on this metric in Figure 4 & Table 2 .

Here we choose training a smaller ResNet18 with pre-activation (He et al., 2016a) on CIFAR-10 as the case study (the test accuracy is reported in Appendix A.5).

For AM2-SGD, we only give the expected convergence results as follows.

Theorem 2.

For the reformulated AM2-SGD, if we choose

Remark: In comparison with Theorem 1a, Theorem 2 has an additional term F (x 0 ) ??? F (x ) in the upper bound, which is inevitable.

This difference comes from different restrictions on the choice of m. For AM2-SGD, m ??? 1 is the only requirement.

Since it is impossible to let m K to obtain an improved rate, this additional term is inevitable.

As a sanity check, we can let m ??? ??? to obtain a point table with almost all x 0 , and then the upper bound becomes exactly F (x 0 ) ??? F (x ).

In some cases, there exists an optimal choice of m > 1 in Theorem 2.

However, the optimal choice could be messy and thus we omit the discussion here.

Understanding: Comparing the rates, we see that when using the same m, AM2-SGD has slightly better dependence on ??, which is related to the observation in Figure 5 that AM2-SGD is always slightly faster than AM1-SGD.

This difference is suggesting that randomly incorporating past iterates beyond m iterations helps.

If m = O(1), Theorems 1 and 2 establish the optimal O(L/K 2 + (?? + M )/ ??? K) rate in the convex setting (see Lan (2012) for optimality), which verifies AM1-SGD and AM2-SGD as variants of the Nesterov's method (Nesterov, 1983; 2013b) .

From the above analysis, the effect of m can be understood as trading acceleration for variance control.

However, since both acceleration and variance control boost the convergence speed, the reduced final performance observed in the CIFAR experiments may not always be the case as will be shown in Figure 5 and Table 3 .

Connections with Katyusha Our original inspiration of AM1-SGD comes from the construction of Katyusha (Allen-Zhu, 2018), the recent breakthrough in finite-sum convex optimization, which uses a previously calculated "snapshot" point to provide momentum, i.e., Katyusha momentum.

AM1-SGD also uses an aggregated point to provide momentum and it shares many structural similarities with Katyusha.

We refer the interested readers to Appendix B.3.

In this section, we evaluate AM1-SGD and AM2-SGD on more deep learning tasks.

Our goal is to show their potentials of serving as alternatives for M-SGD.

Regarding the options: for AM1-SGD, Option I is a nice choice, which has slightly better final performance as shown in Table 1 ; for AM2-SGD, Option I is not recommended as mentioned before.

Here we choose to evaluate Option II for both methods for consistency, which also corresponds to the analysis in Section 4.

AM1-SGD and AM2-SGD use exactly the same values for (??, ??) as M-SGD, which was tuned to optimize the performance of M-SGD.

We set m = 5 for AM1-SGD and AM2-SGD.

We trained ResNet50 and ResNet152 (He et al., 2016b) on the ILSVRC2012 dataset ("ImageNet") (Russakovsky et al., 2015) shown in Figure 5b .

For this task, we used 0.1 initial learning rate and 0.9 momentum for all methods, which is a typical choice.

We performed a restart after each learning rate reduction as discussed in Appendix A.4.

We believe that this helps the training process and also does not incur any additional overhead.

We report the final accuracy in Table 3 .

We also did a language model experiment on Penn Treebank dataset (Marcus et al., 1993) .

We used the LSTM (Hochreiter & Schmidhuber, 1997) model defined in Merity et al. (2017) and followed the experimental setup in its released code.

We only changed the learning rate and momentum in (Polyak & Juditsky, 1992) with constant learning rate 30 as used in Merity et al. (2017) .

For the choice of (??, ??), following Lucas et al. (2019), we chose ?? = 0.99 and used the scheduler that reduces the learning rate by half when the validation loss has not decreased for 15 epochs.

We swept ?? from {5, 2.5, 1, 0.1, 0.01} and found that ?? = 2.5 resulted in the lowest validation perplexity for M-SGD.

We thus ran AM1-SGD and AM2-SGD with this (??, ??) and m = 5.

Due to the small decay factor, we did not restart AM1-SGD and AM2-SGD after learning rate reductions.

The validation perplexity curve is plotted in Figure 5a .

We report validation perplexity and test perplexity in Table 3 .

This experiment is directly comparable with the one in Lucas et al. (2019) .

Extra results are provided in the appendices for interested readers: the robustness when using large ?? (Appendix A.2), a CIFAR-100 experiment (Appendix A.6) and comparison with classical momentum (Polyak, 1964)

We presented Amortized Nesterov's Momentum, which is a special variant of Nesterov's momentum that utilizes several past iterates to provide the momentum.

Based on this idea, we designed two different realizations, namely, AM1-SGD and AM2-SGD.

Both of them are simple to implement with little-to-no additional tuning overhead over M-SGD.

Our empirical results demonstrate that switching to AM1-SGD and AM2-SGD produces faster early convergence and comparable final generalization performance.

AM1-SGD is lightweight and has more robust iterates than M-SGD, and thus can serve as a favorable alternative to M-SGD in large-scale deep learning tasks.

AM2-SGD could be favorable for more restrictive tasks (e.g., asynchronous training) due to its extensibility and good performance.

Both the methods are proved optimal in the convex case, just like M-SGD.

Based on the intuition from convex analysis, the proposed methods are trading acceleration for variance control, which provides hints for the hyper-parameter tuning.

We discuss the issues with learning rate schedulers in Appendix A.4.

We report the test accuracy results of the ResNet18 experiment (in Section 4) in Appendix A.5.

A CIFAR-100 experiment is provided in Appendix A.6.

We also provide a sanity check for our implementation in Appendix A.7.

Table 4 : Final test accuracy and average accuracy STD of training ResNet34 on CIFAR-10 over 5 runs (including the detailed data of the curves in Figure 1 and Figure 2a) .

For all the methods, ?? 0 = 0.1, ?? = 0.9.

Multiple runs start with the same x 0 .

We show in Figure 6 how m affects the convergence of test accuracy.

The results show that increasing m speeds up the convergence in the early stage.

While for AM1-SGD the convergences of Option I and Option II are similar, AM2-SGD with Option II is consistently better than with Option I in this experiment.

It seems that AM2-SGD with Option I does not benefit from increasing m and the algorithm is not robust.

Thus, we do not recommend using Option I for AM2-SGD.

Table 4 .

Labels are formatted as 'AM1/2-SGD-{Option}-{m}' .

Best viewed in color.

We compare the robustness of M-SGD and AM1-SGD when ?? is large in Figure 7 & Table 5 .

For fair comparison, AM1-SGD uses Option I. As we can see, the STD error of M-SGD scales up significantly when ?? is larger and the performance is more affected by a large ?? compared with AM1-SGD.

CM-SGD with its typical hyper-parameter settings (?? 0 = 0.1, ?? = 0.9) is observed to achieve similar generalization performance as M-SGD.

However, CM-SGD is more unstable and prone to oscillations (Lucas et al., 2019) , which makes it less robust than M-SGD as shown in Table 6 .

Aggregated Momentum (AggMo) AggMo combines multiple momentum buffers, which is inspired by the passive damping from physics literature (Lucas et al., 2019) .

AggMo uses the following update rules (for t = 1, . . .

, T ,

We used the exponential hyper-parameter setting recommended in the original work with the scalefactor a = 0.1 fixed, ?? (t) = 1 ??? a t???1 , for t = 1, . . .

, T and choosing T in {2, 3, 4}. We found that T = 2 gave the best performance in this experiment.

As shown in Figure 8 & Table 6 , with the help of passive damping, AggMo is more stable and robust compared with CM-SGD. (2019) introduce the immediate discount factor ?? ??? R for the momentum scheme, which results in the QHM update rules (?? ??? R,

Here we used the recommended hyper-parameter setting for QHM (?? 0 = 1.0, ?? = 0.999, ?? = 0.7).

Figure 8 shows that AM1-SGD, AggMo and QHM achieve faster convergence in the early stage while CM-SGD has the highest final accuracy.

In terms of robustness, huge gaps are observed when comparing AM1-SGD with the remaining methods in Table 6 .

Note that AM1-SGD is more efficient than both QHM and AggMo, and is as efficient as CM-SGD.

We also plot the convergence of train-batch loss for all the methods in Figure 9 .

Despite of showing worse generalization performance, both QHM and AggMo perform better on reducing the trainbatch loss in this experiment, which is consistent with the results reported in Ma & Yarats (2019) We show in Figure 10 that when ?? is large for the task, using step learning rate scheduler with decay factor 10, a performance drop is observed after each reduction.

Both Option I and Option II have this issue and the curves are basically identical.

Here we only use Option II.

We fix this issue by performing a restart after each learning rate reduction (labeled with '+').

We plot the train-batch loss here because we find the phenomenon is clearer in this way.

If ?? = 0.9, there is no observable performance drop in this experiment.

For smooth-changing schedulers such as the cosine annealing scheduler (Loshchilov & Hutter, 2016) , the amortized momentum works well as shown in Figure 11 .

We report the test accuracy results of the experiments in Section 4 in Figure 12 & Table 7 Table 7 : ResNet18 with pre-activation on CIFAR-10.

For all methods, ?? 0 = 0.1, ?? = 0.9, run 20 seeds.

For AM1-SGD, m = 5 and its labels are formatted as 'AM1-SGD-{Option}'.

Shaded bands indicate ??1 standard deviation.

Best viewed in color.

We report the results of training DenseNet121 (Huang et al., 2017) on CIFAR-100 in Figure 13 , which shows that both AM1-SGD and AM2-SGD perform well before the final learning rate reduction.

However, the final accuracies are lowered around 0.6% compared with M-SGD.

We also notice that SGD reduces the train-batch loss at an incredibly fast rate and the losses it reaches are consistently lower than other methods in the entire 300 epochs.

However, this performance is not reflected in the convergence of test accuracy.

We believe that this phenomenon suggests that the DenseNet model is actually "overfitting" M-SGD (since in the ResNet experiments, M-SGD always achieves a lower train loss than SGD after the final learning rate reduction).

A.7 A SANITY CHECK When m = 1, both AM1-SGD and AM2-SGD are equivalent to M-SGD, we plot their convergence in Figure 14 as a sanity check (the detailed data is given in Table 4 ).

We observed that when m = 1, both AM1-SGD and AM2-SGD have a lower STD error than M-SGD.

We believe that it is because they both maintain the iterates without scaling, which is numerically more stable than M-SGD (M-SGD in standard PyTorch maintains a scaled buffer, i.e., v

When h ??? 0 and ?? is a constant, we do the reformulations by eliminating the sequence {z k }.

For the reformulated AM2-SGD,

The reformulated AM2-SGD

For the reformulated AM1-SGD, when h ??? 0, the inner loops are basically SGD,

At the end of each inner loop (i.e., when (k + 1) mod m = 0), we have

while at the beginning of the next inner loop,

which means that we need to set x k+1 ??? x k+1 + ?? ?? (x s+1 ???x s ) (reassign the value of x k+1 ).

We also give the reformulation of M-SGD (scheme (1)) to the Auslender & Teboulle (2006) scheme for reference:

Auslender & Teboulle (2006) (AC-SA (Lan, 2012))

Nesterov (

Intuition for the Auslender & Teboulle (2006) scheme can be found in Remark 2 in Lan (2012).

The reformulated schemes are copied here for reference:

Initialize:

for j = 0, . . .

, m ??? 1 do 3:

4:

5:

end for 8:x s+1 = 1 m m j=1 x sm+j .

9: end for Output:x S .

Sample j k uniformly in [m].

3:

Comparing the reformulated schemes, we see that their iterations can be generalized as follows:

This type of scheme is first proposed in Auslender & Teboulle (2006), which represents one of the simplest variants of the Nesterov's methods (see Tseng (2008) for other variants).

The scheme is then modified into various settings (Hu et al., 2009; Lan, 2012; Ghadimi & Lan, 2012; 2016; Zhou et al., 2019; Lan et al., 2019 ) to achieve acceleration.

The following lemma serves as a cornerstone for the convergence proofs of AM1-SGD and AM2-SGD.

Lemma 1.

If ??(1 ??? ??) < 1/L, the update scheme (6) satisfies the following recursion:

This Lemma is similarly provided in Lan (2012); Ghadimi & Lan (2012) under a more general setting that allows non-Euclidean norms in the assumptions, we give a proof here for completeness.

Based on the convexity (Assumption (a)), we have

We upper bound the terms on the right side one-by-one.

where ( ) uses the relation between x and z, i.e.,

For R 2 , based on Assumption (a), we have

Then, noting that x ??? y + = (1 ??? ??) ?? (z ??? z + ), we can arrange the above inequality as

Using Young's inequality with ?? > 0, we obtain

For R 3 , based on the optimality condition of prox ??h {z ??? ?? ?? ???f i (x)} and denoting ???h(z + ) as a subgradient of h at z + , we have for any u ??? X,

where

Finally, by upper bounding (7) using (8), (9), (10), we conclude that

Note that with the convexity of h and y

Using the above inequality and choosing

Using Assumption (c), Lemma 1 with

and taking expectation, if ?? s (1 ??? ?? s ) < 1/L, we have

Summing the above inequality from k = sm, . . .

, sm + m ??? 1, we obtain

Using the definition ofx s+1 and convexity,

It can be verified that with the choices ?? s = s s+2 and ?? s = ??1 L(1?????s) , the following holds for s ??? 0,

Note that since our analysis aims at providing intuition, we do not refine the choice of ?? s as in (Hu et al., 2009; Ghadimi & Lan, 2012) .

Thus, by telescoping (13) from s = S ??? 1, . . .

, 0, we obtain

, and thus,

Under review as a conference paper at ICLR 2020 where (a) follows from ?? 1 ??? 2 3 and (b) holds because 0 ??? x ??? (x + 2) 2 is non-decreasing and thus

, and based on the choice of ?? 1 = min

Thus, we conclude that

Substituting S = K/m completes the proof.

In order to prove Theorem 1b, we need the following known result for the martingale difference (cf.

Lemma 2 in Lan et al. (2012)):

Summing the above inequality from k = sm, . . .

, sm + m ??? 1 and using the choice ?? s = ??1 L(1?????s)

With our parameter choices, the relations in (14) hold and thus we can telescope the above inequality from s = S ??? 1, . . .

, 0,

Denoting

where ( ) uses the additional assumption

Then, based on Markov's inequality, we have for any ?? ??? 0,

For R 5 , since we have

which is based on the "light tail" assumption, using Lemma 2, we obtain

Combining (15), (16) and (17), based on the parameter setting (cf.

(5)) and using the notation

we conclude that

For R 6 , using the choice of ?? s and ?? 1 , we obtain

which completes the proof.

Using Assumption (c), Lemma 1 with

Note that

Dividing both sides of (18) by m and then adding

to both sides, we obtain

.

Under review as a conference paper at ICLR 2020 B.3 CONNECTIONS BETWEEN AM1-SGD AND KATYUSHA

The discussion in this section aims to shed light on the understanding of the experimental results, which also shows some interesting relations between AM1-SGD and Katyusha.

The high level idea of Katyusha momentum is that it works as a "magnet" inside an epoch of SVRG updates, which "stabilizes" the iterates so as to make Nesterov's momentum effective (Allen-Zhu, 2018) .

In theory, the key effect of Katyusha momentum is that it allows the tightest possible variance bound for the stochastic gradient estimator of SVRG (cf.

Lemma 2.4 and its comments in AllenZhu (2018)).

In this sense, we can interpret Katyusha momentum as a variance reducer that further reduces the variance of SVRG.

Below we show the similarity between the construction of Katyusha and AM1-SGD, based on which we conjecture that the amortized momentum can also reduce the variance of SGD (and thus increase the robustness).

However, in theory, following a similar analysis of Katyusha, we cannot guarantee a reduction of ?? in the worst case.

Deriving AM1-SGD from Katyusha Katyusha has the following scheme (non-proximal, in the original notations, ?? is the strong convexity parameter, cf.

Algorithm 1 with Option I in Allen-Zhu (2018)) 12 :

Initialize:

Compute and store ???f ( x s ).

for j = 0, . . .

, m ??? 1 do 4:

5:

7:

8:

end for 10:

11: end for Output:

We can eliminate the sequence {z k } in this scheme.

Note that in the parameter setting of Katyusha, we have ?? = ???? 1 , and thus

Hence, the inner loops can be written as

which is the Nesterov's scheme (scheme (1)).

At the end of each inner loop (when k = sm+m???1),

while at the beginning of the next inner loop,

which means that we need to set

Then, the following is an equivalent scheme of Katyusha: 12 We change the notation x k+1 to x k .

Initialize:

for j = 0, . . .

, m ??? 1 do 3:

5:

end for 7:

8:

Now it is clear that the inner loops use Nesterov's momentum and the Katyusha momentum is injected for every m iterations.

If we replace the SVRG estimator ??? k with ???f i k (x k ), set 1 ??? ?? 1 ??? ?? 2 = 0, which is to eliminate Nesterov's momentum, and use a uniform average for x s+1 , the above scheme becomes exactly AM1-SGD (Algorithm 1).

If we only replace the SVRG estimator ??? k , the scheme can be regarded as adding amortized momentum to M-SGD.

This scheme requires tuning the ratio of Nesterov's momentum and amortized momentum.

In our preliminary experiments, after suitable tuning, we observed some performance improvement.

However, this scheme increases the complexity, which we do not consider it worthwhile.

A recent work (Zhou et al., 2018) shows that when 1 ??? ?? 1 ??? ?? 2 = 0, which is to solely use Katyusha momentum, one can still derive optimal rates and the algorithm is greatly simplified.

Their proposed algorithm (i.e., MiG) is structurally more similar to AM1-SGD.

This scheme is equivalent to the PyTorch formulation (scheme (3)) through v Based on this formulation, ?? is understood as the effective learning rate (i.e., the vector it scales has the same cardinality as a gradient) and the experiments in Ma & Yarats (2019) were conducted with fixed ?? = 1.

Their results indicate that when using the same effective learning rate, M-SGD and SGD achieve similar performance and thus they suspect that the benefit of momentum basically comes from using sensible learning rates.

Here we provide some intuition on their results based on convex analysis.

For simplicity, we consider deterministic smooth convex optimization.

In theory, to obtain the optimal convergence rate, the effective learning rate ?? is set to a very large O(k/L), which can be derived from Theorem 1 or Theorem 2 by setting ?? = 0, M = 0, m = 1 (then ?? 1 or ?? 2 is always 2 3 since the other term is ???).

If we fix ?? = 2 3L for both methods, GD has an O(1/K) convergence rate (cf.

Theorem 2.1.13 in Nesterov (2013b)).

For the Nesterov's method, if we use ?? k = k k+2 , it has the convergence rate (applying Lemma 1): Thus, in this case, both GD and the Nesterov's method yield an O(1/K) rate, and thus we expect them to have similar performance.

This analysis suggests that the acceleration effect basically comes from choosing a large effective learning rate, which corresponds to the observations in Ma & Yarats (2019) .

However, what is special about the Nesterov's method is that it finds a legal way to adopt a large ?? that breaks the 1/L limitation.

If GD uses the same large ??, we would expect it to be unstable and potentially diverge.

In this sense, Nesterov's momentum can be understood as a "stabilizer".

In our basic case study (ResNet34 on CIFAR-10), if we align the effective learning rate and set ?? = 1.0 for SGD, the final accuracy is improved but the performance is highly unstable and not robust, which is 2.205% average STD of test accuracy over 5 runs.

The significance of QHM (Ma & Yarats, 2019) is that with suitable tuning, it achieves much faster convergence without changing the effective learning rate.

Our work uses the convergence behavior of SGD as a reference to reflect and to understand the features of our proposed momentum, which is why we set ?? = ??.

0.5 probability.

We used step (or multi-step) learning rate scheduler with a decay factor 10.

For the CIFAR-10 experiments, we trained 90 epochs and decayed the learning rate every 30 epochs.

For the CIFAR-100 experiments, we trained 300 epochs and decayed the learning rate at 150 epoch and 225 epoch following the settings in DenseNet (Huang et al., 2017) .

In the ImageNet experiments, we tried both ResNet50 and ResNet152 (He et al., 2016b) .

The training strategy is the same as the PyTorch's official repository https://github.

com/pytorch/examples/tree/master/imagenet, which uses a batch size of 256.

The learning rate starts at 0.1 and decays by a factor of 10 every 30 epochs.

Also, we applied weight decay with 0.0001 decay rate to the model during the training.

For the data augmentation, we applied random 224-pixel crops and random horizontal flips with 0.5 probability.

Here, we run all experiments across 8 NVIDIA P100 GPUs for 90 epochs.

We followed the implementation in the repository https://github.com/salesforce/ awd-lstm-lm and trained word level Penn Treebank with LSTM without fine-tuning or continuous cache pointer augmentation for 750 epochs.

The experiments were conducted on a single RTX2080Ti.

We used the default hyper-parameter tuning except for learning rate and momentum: The LSTM has 3 layers containing 1150 hidden units each, embedding size is 400, gradient clipping has a maximum norm 0.25, batch size is 80, using variable sequence length, dropout for the layers has probability 0.4, dropout for the RNN layers has probability 0.3, dropout for the input embedding layer has probability 0.65, dropout to remove words from embedding layer has probability 0.1, weight drop (Merity et al., 2017) has probability 0.5, the amount of 2-regularization on the RNN activation is 2.0, the amount of slowness regularization applied on the RNN activation is 1.0 and all weights receive a weight decay of 0.0000012.

<|TLDR|>

@highlight

Amortizing Nesterov's momentum for more robust, lightweight and fast deep learning training.