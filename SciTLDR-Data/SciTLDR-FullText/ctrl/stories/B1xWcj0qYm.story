Empirical risk minimization (ERM), with proper loss function and regularization, is the common practice of supervised classification.

In this paper, we study training arbitrary (from linear to deep) binary classifier from only unlabeled (U) data by ERM.

We prove that it is impossible to estimate the risk of an arbitrary binary classifier in an unbiased manner given a single set of U data, but it becomes possible given two sets of U data with different class priors.

These two facts answer a fundamental question---what the minimal supervision is for training any binary classifier from only U data.

Following these findings, we propose an ERM-based learning method from two sets of U data, and then prove it is consistent.

Experiments demonstrate the proposed method could train deep models and outperform state-of-the-art methods for learning from two sets of U data.

With some properly chosen loss function (e.g., BID2 BID50 BID39 and regularization (e.g., BID51 BID46 , empirical risk minimization (ERM) is the common practice of supervised classification BID54 .

Actually, ERM is used in not only supervised learning but also weakly-supervised learning.

For example, in semi-supervised learning (Chapelle et al., 2006) , we have very limited labeled (L) data and a lot of unlabeled (U) data, where L data share the same form with supervised learning.

Thus, it is easy to estimate the risk from only L data in order to carry out ERM, and U data are needed exclusively in regularization (including but not limited to BID16 BID3 BID25 BID29 BID20 BID49 BID24 Kamnitsas et al., 2018) .Nevertheless, L data may differ from supervised learning in not only the amount but also the form.

For instance, in positive-unlabeled learning BID12 BID55 ), all L data are from the positive class, and due to the lack of L data from the negative class it becomes impossible to estimate the risk from only L data.

To this end, a two-step approach to ERM has been considered (du BID9 BID34 BID18 .

Firstly, the risk is rewritten into an equivalent expression, such that it just involves the same distributions from which L and U data are sampled-this step leads to certain risk estimators.

Secondly, the risk is estimated from both L and U data, and the resulted empirical training risk is minimized (e.g. by BID41 Kingma & Ba, 2015) .

In this two-step approach, U data are needed absolutely in ERM itself.

This indicates that risk rewrite (i.e., the technique of making the risk estimable from observable data via an equivalent expression) enables ERM in positive-unlabeled learning and is the key of success.

One step further from positive-unlabeled learning is learning from only U data without any L data.

This is significantly harder than previous learning problems (cf.

FIG1 ).

However, we would still like to train arbitrary binary classifier, in particular, deep networks BID15 .

Note that for this purpose clustering is suboptimal for two major reasons.

First, successful translation of clusters into meaningful classes completely relies on the critical assumption that one cluster exactly In the left panel, (a) and (b) show positive (P) and negative (N) components of the Gaussian mixture; (c) and (d) show two distributions (with class priors 0.9 and 0.4) where U training data are drawn (marked as black points).

The right panel shows the test distribution (with class prior 0.3) and data (marked as blue for P and red for N), as well as four learned classifiers.

In the legend, "CCN" refers to BID31 , "UU-biased" means supervised learning taking larger-/smaller-class-prior U data as P/N data, "UU" is the proposed method, and "Oracle" means supervised learning from the same amount of L data.

See Appendix B for more information.

We can see that UU is almost identical to Oracle and much better than the other two methods.

corresponds to one class, and hence even perfect clustering might still result in poor classification.

Second, clustering must introduce additional geometric or information-theoretic assumptions upon which the learning objectives of clustering are built (e.g., BID57 BID14 .

As a consequence, we prefer ERM to clustering and then no more assumption is required.

The difficulty is how to estimate the risk from only U data, and our solution is again ERM-enabling risk rewrite in the aforementioned two-step approach.

The first step should lead to an unbiased risk estimator that will be used in the second step.

Subsequently, we can evaluate the empirical training and/or validation risk by plugging only U training/validation data into the risk estimator.

Thus, this two-step ERM needs no L validation data for hyperparameter tuning, which is a huge advantage in training deep models nowadays.

Note that given only U data, by no means could we learn the class priors BID27 , so that we assume all necessary class priors are also given.

This is the unique type of supervision we will leverage throughout this paper, and hence this learning problem still belongs to weakly-supervised learning rather than unsupervised learning.

In this paper, we raise a fundamental question in weakly-supervised learning-how many sets of U data with different class priors are necessary for rewriting the risk?

Our answer has two aspects:??? Risk rewrite is impossible given a single set of U data (see Theorem 2 in Sec. 3);??? Risk rewrite becomes possible given two sets of U data (see Theorem 4 in Sec. 4).This suggests that three class priors 1 are all you need to train deep models from only U data, while any two 2 should not be enough.

The impossibility is a proof by contradiction, and the possibility is a proof by construction, following which we explicitly design an unbiased risk estimator.

Therefore, with the help of this risk estimator, we propose an ERM-based learning method from two sets of U data.

Thanks to the unbiasedness of our risk estimator, we derive an estimation error bound which certainly guarantees the consistency of learning BID30 BID44 .3 Experiments demonstrate that the proposed method could train multilayer perceptron, AllConvNet BID45 and ResNet (He et al., 2016) from two sets of U data; it could outperform state-of-the-art methods for learning from two sets of U data.

See FIG1 for how the proposed method works on a Gaussian mixture of two components.

As mentioned earlier, learning from two sets of U data is already studied in du BID8 and BID27 .

Both of them adopt (4) as the performance measure.

In the former paper, g is learned by estimating sign(p tr (x) ??? p tr (x)).

In the latter paper, g is learned by taking noisy L data from p tr (x) and p tr (x) as clean L data from p p (x) and p n (x), and then its threshold is moved to the correct value by post-processing.

In summary, instead of ERM, they evidence the possibility of empirical balanced risk minimization, and no impossibility is proven.

Our findings are compatible with learning from label proportions BID36 BID58 .

BID36 proves that the minimal number of U sets is equal to the number of classes.

However, their finding only holds for the linear model, the logistic loss, and their proposed method based on mean operators.

On the other hand, BID58 is not ERM-based; it is based on discriminative clustering together with expectation regularization BID25 .At first glance, our data generation process, using the names from BID27 , looks quite similar to class-conditional noise (CCN, BID0 in learning with noisy labels (cf.

BID31 .

4 In fact, BID27 makes use of mutually contaminated distributions (MCD, BID43 that is more general than CCN.

Denote by??? andp(??) the corrupted label and distributions.

Then, CCN and MCD are defined by DISPLAYFORM0 where both of T CCN and T MCD are 2-by-2 matrices but T CCN is column normalized and T MCD is row normalized.

It has been proven in BID27 that CCN is a strict special case of MCD.

To DISPLAYFORM1 Due to this covariate shift, CCN methods do not fit MCD problem setting, though MCD methods fit CCN problem setting.

To the best of our knowledge, the proposed method is the first MCD method based on ERM.3 LEARNING FROM ONE SET OF U DATA From now on, we prove that knowing ?? p and ?? is insufficient for rewriting R(g).

To begin with, we review ERM BID54 by imaging that we are given X p = {x 1 , . . .

, x n } ??? p p (x) and X n = {x 1 , . . . , x n } ??? p n (x).

Then, we would go through the following procedure: DISPLAYFORM0 4.

Minimize R pn (g), with appropriate regularization, by favorite optimization algorithm.

Here, should be classification-calibrated BID2 , 5 in order to guarantee that R(g; ) and R(g; 01 ) have the same minimizer over all measurable functions.

This minimizer is the Bayes optimal classifier and denoted by g * * = arg min g R(g).

The Bayes optimal risk R(g * * ) is usually unachievable by ERM as n, n ??? ???. That is why by choosing a model G, g * = arg min g???G R(g) became the target (i.e., g pn = arg min g???G R pn (g) will converge to g * as n, n ??? ???).

In statistical learning, the approximation error is R(g * ) ??? R(g * * ), and the estimation error is R( g pn ) ??? R(g * ).

Learning is consistent if and only if the estimation error converges to zero as n, n ??? ???.

Recall that R(g) is approximated by (5) given X p and X n , which does not work given X tr and X tr .

We might rewrite R(g) so that it could be approximated given X tr and/or X tr .

This is known as the backward correction in learning with noisy/corrupted labels BID35 ; see also BID31 BID53 .

Definition 1.

We say that R(g) in (3) is rewritable given p tr , if and only if 6 there exist constants a and b, such that for any g it holds that DISPLAYFORM0 where DISPLAYFORM1 is the corrected loss function.

In Eq. (6), the expectation is with respect to p tr and ?? is a free variable in it.

The impossibility will be stronger, if ?? is unspecified and allowed to be adjusted according to ?? p .Theorem 2.

Let be 01 , or any bounded surrogate loss satisfying that DISPLAYFORM2 Assume p p and p n are almost surely separable.

Then, R(g) is not rewritable though ?? is free.

This theorem shows that under the separability assumption of p p and p n , R(g) is not rewritable.

As a consequence, we lack a learning objective, that is, the empirical training risk.

It is even worse-we cannot access the empirical validation risk of g after it is trained by other learning methods such as discriminative clustering.

In particular, 01 satisfies FORMULA5 , which implies that the common practice of hyperparameter tuning is disabled by Theorem 2, since U validation data are also drawn from p tr .

From now on, we prove that knowing ?? p , ?? and ?? is sufficient for rewriting R(g).

We have proven that R(g) is not rewritable given p tr , and BID36 has proven that R(g) can be estimated from X tr and X tr , where g is a linear model and is the logistic loss.

These facts motivate us to investigate the possibility of rewriting R(g), where g and are both arbitrary.

Definition 3.

We say that R(g) is rewritable given p tr and p tr , if and only if 9 there exist constants a, b, c and d, such that for any g it holds that DISPLAYFORM0 DISPLAYFORM1 are the corrected loss functions.

In Eq. FORMULA6 , the expectations are with respect to p tr and p tr that are regarded as the corrupted p p and p n .

There are two free variables ?? and ?? in p tr and p tr .

The possibility will be stronger, if ?? and ?? are already specified and disallowed to be adjusted according to ?? p .Theorem 4.

Fix ?? and ?? .

Assume ?? > ?? ; otherwise, swap p tr and p tr to make sure ?? > ?? .

Then, R(g) is rewritable, by letting DISPLAYFORM2 Theorem (4) immediately leads to an unbiased risk estimator, namely, DISPLAYFORM3 Eq. FORMULA9 is useful for both training (by plugging U training data into it) and hyperparameter tuning (by plugging U validation data into it).

We hereafter refer to the process of obtaining the empirical risk minimizer of (10), i.e., g uu = arg min g???G R uu (g), as unlabeled-unlabeled (UU) learning.

The proposed UU learning is by nature ERM-based, and consequently g uu can be obtained by powerful stochastic optimization algorithms (e.g., BID11 Kingma & Ba, 2015) .Simplification Note that (10) may require some efforts to implement.

Fortunately, it can be simplified by employing that satisfies a symmetric condition: DISPLAYFORM4 Eq. (11) covers 01 , a ramp loss ramp (z) = max{0, min{1, (1 ??? z)/2}} in du Plessis et al. FORMULA20 and a sigmoid loss sig (z) = 1/(1 + exp(z)) in BID18 .

With the help of FORMULA10 , FORMULA9 can be simplified as DISPLAYFORM5 where FORMULA11 is an unbiased risk estimator, and it is easy to implement with existing codes of cost-sensitive learning.

DISPLAYFORM6 Special cases Consider some special cases of (10) by specifying ?? and ?? .

It is obvious that (10) reduces to (5) for supervised learning, if ?? = 1 and ?? = 0.

Next, (10) reduces to DISPLAYFORM7 if ?? = 1 and ?? = ?? p , and we recover the unbiased risk estimator in positive-unlabeled learning BID10 BID18 .

Additionally, (10) reduces to a fairly complicated unbiased risk estimator in similar-unlabeled learning BID1 , if ?? = ?? p , ?? = ?? 2 p /(2?? 2 p ??? 2?? p + 1) or vice versa.

Therefore, UU learning is a very general framework in weakly-supervised learning.

The consistency of UU learning is guaranteed due to the unbiasedness of (10).

In what follows, we analyze the estimation error R( g uu ) ??? R(g * ) (see Sec. 3.1 for the definition).

To this end, assume there are C g > 0 and C > 0 such that sup g???G g ??? ??? C g and sup |z|???Cg (z) ??? C , and assume (z) is Lipschitz continuous for all |z| ??? C g with a Lipschitz constant L .

Let R n (G) and R n (G) be the Rademacher complexity of G over p tr (x) and p tr (x) BID30 BID44 .

For convenience, denote by ?? n,n = ??/ ??? n + ?? / ??? n .Theorem 5.

For any ?? > 0, let C ?? = (ln 2/??)/2, then we have with probability at least 1 ??? ??, DISPLAYFORM0 where the probability is over repeated sampling of X tr and X tr for training g uu .Theorem 5 ensures that UU learning is consistent (and so are all the special cases): as n, n ??? ???, DISPLAYFORM1 for all parametric models with a bounded norm such as deep networks trained with weight decay.

Moreover, DISPLAYFORM2 where O p denotes the order in probability, for all linear-in-parameter models with a bounded norm, including non-parametric kernel models in reproducing kernel Hilbert spaces BID42 .

In this section, we experimentally analyze the proposed method in training deep networks and subsequently experimentally compare it with state-of-the-art methods for learning from two sets of U data.

The implementation in our experiments is based on Keras (see https://keras.io); it is available at https://github.com/lunanbit/UUlearning.

In order to analyze the proposed method, we compare it with three supervised baseline methods:??? small PN means supervised learning from 10% L data;??? PN oracle means supervised learning from 100% L data;??? small PN prior-shift means supervised learning from 10% L data under class-prior change.

Notice that the first two baselines have L data identically distributed as the test data, which is very advantageous and thus the experiments in this subsection are merely for a proof of concept.

TAB0 summarizes the benchmarks.

They are converted into binary classification datasets; please see Appendix C.1 for details.

X tr and X tr of the same sample size are drawn according to Eq. (1) , where ?? and ?? are chosen as 0.9, 0.1 or 0.8, 0.2.

The test data are just drawn from p(x, y).

TAB0 also describes the models and optimizers.

In this table, FC refers to fully connected neural networks, AllConvNet refers to all convolutional net BID45 and ResNet refers to residual networks (He et al., 2016); then, SGD is short for stochastic gradient descent BID41 and Adam is short for adaptive moment estimation (Kingma & Ba, 2015) .Recall from Sec. 3.1 that after the model and optimizer are chosen, it remains to determine the loss (z).

We have compared the sigmoid loss sig (z) and the logistic loss log (z) = ln(1 + exp(???z)), and found that the resulted classification errors are similar; please find the details in Appendix C.2.

Since sig satisfies (11) and is compatible with (12), we shall adopt it as the surrogate loss.

The experimental results are reported in FIG2 , where means and standard deviations of classification errors based on 10 random samplings are shown, and the table of final errors can be found in Appendix C.2.

When ?? = 0.9 and ?? = 0.1 (cf.

the left column), UU is comparable to PN oracle in most cases.

When ?? = 0.8 and ?? = 0.2 (cf.

the right column), UU performs slightly worse but it is still better than small PN baselines.

This is because the task becomes harder when ?? and ?? become closer, which will be intensively investigated next.

On the closeness of ?? and ?? It is intuitive that if ?? and ?? move closer, X tr and X tr will be more similar and thus less informative.

To investigate this, we test UU and CCN BID31 on MNIST by fixing ?? to 0.9 or 0.8 and gradually moving ?? from 0.1 to 0.5, and the experimental results are reported in FIG3 .

We can see that when ?? moves closer to ??, UU and CCN become worse, while UU is affected slightly and CCN is affected severely.

The phenomenon of UU can be explained by Theorem 5, where the upper bound in (13) is linear in ?? and ?? which, as ?? ??? ??, are inversely proportional to ?? ??? ?? .

On the other hand, the phenomenon of CCN is caused by stronger covariate shift when ?? moves closer to ?? rather than the difficulty of the task.

This illustrates CCN methods do not fit our problem setting, so that we called for some new learning method (i.e., UU).

Note that there would be strong covariate shift not only by changing ?? and ?? but also by changing n and n .

The investigation of this issue is deferred to Appendix C.2 due to limited space.

Robustness against inaccurate training class priors Hitherto, we have assumed that the values of ?? and ?? are accessible, which is rarely satisfied in practice.

Fortunately, UU is a robust learning method against inaccurate training class priors.

To show this, let and be real numbers around 1, ?? = ?? and ?? = ?? be perturbed ?? and ?? , and we test UU on MNIST and CIFAR-10 by drawing data using ?? and ?? but training models using ?? and ?? instead.

The experimental results in TAB1 imply that UU is fairly robust to inaccurate ?? and ?? and can be safely applied in the wild.

Finally, we compare UU with two state-of-the-art methods for dealing with two sets of U data:

??? proportion-SVM (pSVM, BID58 that is the best in learning from label proportions;??? balanced error minimization (BER, BID27 that is the most related work to UU.The original codes of BER train single-hidden-layer neural networks by LBFGS (which belongs to second-order optimization) in MATLAB.

For a fair comparison, we also implement BER by fixing ?? p to 0.5 in UU, so that UU and BER only differ in the performance measure.

This new baseline is referred to as BER-FC.

The first five datasets come with the original codes of BER and USPS is from https://cs.nyu.edu/ roweis/data.html.

The rows are arranged according to ??p.

In this table, # Sub means the amount of subsampled L training data, # Train means the amount of generated U training data, and ????? means ?? ??? ?? .

The cell N/A (in MNIST row and pSVM column) is since pSVM is based on maximum margin clustering and is too slow on MNIST.

The task would be harder, if ??p is closer to 0.5, or # Train or ????? is smaller.

The information of datasets can be found in TAB2 .

We work on small datasets following BID27 , because pSVM and BER are not reliant on stochastic optimization and cannot handle larger datasets.

Furthermore, in order to try different ?? p , we first subsample the original datasets to match the desired ?? p and then calculate the sample sizes n and n according to how many P and N data there are in the subsampled datasets, where ?? and ?? are set as close to 0.9 and 0.1 as possible.

For UU and BER-FC, the model is FC with ReLU of depth 5 and the optimizer is SGD.

We repeat this sampling-and-training process 10 times for all learning methods on all datasets.

The experimental results are reported in TAB2 , and we can see that UU is always the best method (7 out of 10 cases) or comparable to the best method (3 out of 10 cases).

Moreover, the closer ?? p is to 0.5, the better BER and BER-FC are; however, the closer ?? p is to 0 or 1, the worse they are, and sometimes they are much worse than pSVM.

This is because their goal is to minimize the balanced error instead of the classification error.

In our experiments, pSVM falls behind, because it is based on discriminative clustering and is also not designed to minimize the classification error.

We focused on training arbitrary binary classifier, ranging from linear to deep models, from only U data by ERM.

We proved that risk rewrite as the core of ERM is impossible given a single set of U data, but it becomes possible given two sets of U data with different class priors, after we assumed that all necessary class priors are also given.

This possibility led to an unbiased risk estimator, and with the help of this risk estimator we proposed UU learning, the first ERM-based learning method from two sets of U data.

Experiments demonstrated that UU learning could successfully train fully connected, all convolutional and residual networks, and it compared favorably with state-of-the-art methods for learning from two sets of U data.

In this appendix, we prove all theorems.

We prove the theorem by contradiction, namely, for any such p(x, y) (with almost surely separable p p and p n ), for all a, b and all ??, we are able to find some g for which (6) fails.

Our argument goes from the special case of 01 to the general case of satisfying (7).Firstly, let g(x) = +??? identically, so that (g(x)) = 0 and (???g(x)) = 1.

Plugging them into (3) and FORMULA3 , we obtain that DISPLAYFORM0 Secondly, let g(x) = ?????? identically; this time (g(x)) = 1 and (???g(x)) = 0, and we obtain that a = ?? p .Thirdly, let g(x) = +??? over p p and g(x) = ?????? over p n .

To be precise, define DISPLAYFORM1 This is possible because g is arbitrary.

The last case g(x) = 0 should have a zero probability, since p p and p n are almost surely separable.

Hence, we have (g(x)) = 0 and (???g(x)) = 1 over p p and (g(x)) = 1 and (???g(x)) = 0 over p n , resulting in DISPLAYFORM2 By solving this equation, we know that DISPLAYFORM3 Nevertheless, 0 ??? ?? ??? 1 whereas DISPLAYFORM4 Therefore, (14) must be a contradiction, unless ?? p = 0 or ?? p = 1 which implies that there is just a single class and the problem under consideration is not binary classification.

Finally, given any satisfying FORMULA5 , it is not difficult to verify that the three g above lead to the same contradiction with exactly the same a, b and ?? by solving a bit more complicated equations.

Let J(g) be an alias of R(g) in Definition 3 serving as the learning objective, i.e., DISPLAYFORM0 then DISPLAYFORM1 On the other hand, DISPLAYFORM2 since J(g) is an alias of R(g).

As a result, in order to minimize R(g) in (3), it suffices to minimize J(g) in FORMULA2 , if we can make DISPLAYFORM3 Solving these equations gives us Eq. FORMULA8 , which concludes the proof.

First, we show the uniform deviation bound, which is useful to derive the estimation error bound.

Lemma 6.

For any ?? > 0, let C ?? = (ln 2/??)/2, then we have with probability at least 1 ??? ??, DISPLAYFORM0 where the probability is over repeated sampling of data for evaluating R uu (g).Proof.

Consider the one-side uniform deviation sup g???G R uu (g) ??? R(g).

Since 0 ??? (z) ??? C , the change of it will be no more than C ??/n if some x i is replaced, or no more than C ?? /n if some x j is replaced.

Subsequently, McDiarmid's inequality BID26 tells us that DISPLAYFORM1 or equivalently, with probability at least 1 ??? ??/2, DISPLAYFORM2 By symmetrization BID54 , it is a routine work to show that DISPLAYFORM3 and according to Talagrand's contraction lemma BID44 , DISPLAYFORM4 The one-side uniform deviation sup g???G R(g) ??? R uu (g) can be bounded similarly.

Based on Lemma 6, the estimation error bound (13) is proven through DISPLAYFORM5 where R uu ( g uu ) ??? R uu (g * ) by the definition of g uu .

In the introduction, we illustrated the learning problem and the proposed method using a Gaussian mixture of two components.

The details of this illustrative example are presented here.

The P component p p (x) and N component p n (x) are both two-dimensional Gaussian distributions.

Their means are DISPLAYFORM0 and their covariance is the identity matrix.

The two training distributions are created following (1) with class priors ?? = 0.9 and ?? = 0.4.

Subsequently, the two sets of U training data were sampled from those distributions with sample sizes n = 2000 and n = 1000.

Moreover, p p (x) and p n (x) are combined to form the test distribution p(x, y) with weights 0.3 and 0.7, so ?? p = 0.3.Note that p(x) changes between training and test distributions (which can be seen from FIG1 by comparing (c) and (d) in the left panel and the right panel).

This is the key difference between UU and CCN BID31 .For training, a linear (-in-input) model g(x) = ?? x + b where ?? ??? R 2 and b ??? R, and a sigmoid loss sig (z) = 1/(1 + exp(z)) were used.

SGD was employed for optimization, where the learning rate was 0.01 and the batch size was 128.

The model just has three parameters, so for the sake of a clear comparison of different risk estimators, we did not add any regularization.

For every method, the model was trained 500 epochs.

The final models are plotted in FIG1 .

MNIST This is a grayscale image dataset of handwritten digits from 0 to 9 where the size of the images is 28*28.

It contains 60,000 training images and 10,000 test images.

Since it has 10 classes originally, we used the even digits as the P class and the odd digits as the N class, respectively.

The model was FC with ReLU as the activation function: d-300-300-300-300-1.

Batch normalization (Ioffe & Szegedy, 2015) was applied before hidden layers.

An 2 -regularization was added, where the regularization parameter was fixed to 1e-4.

The model was trained by SGD with an initial learning rate 1e-3 and a batch size 128.

In addition, the learning rate was decreased by DISPLAYFORM0 where decay was chosen from {0, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4}. This is a learning rate schedule built in Keras.

Fashion-MNIST This is also a grayscale image dataset similarly to MNIST, but here each data is associated with a label from 10 fashion item classes.

It was converted into a binary classification dataset as follows:??? the P class is formed by 'T-shirt', 'Pullover', 'Coat', 'Shirt', and 'Bag';??? the N class is formed by 'Trouser', 'Dress', 'Sandal', 'Sneaker', and 'Ankle boot'.The model and optimizer were same as MNIST, except that the initial learning rate was 1e-4.SVHN This is a 32*32 color image dataset of street view house numbers from 0 to 9.

It consists of 73,257 training data, 26,032 test data, and 531,131 extra training data.

We sampled 100,000 data for training from the concatenation of training data and extra training data-the extra training data were used to ensure enough training data so as to perform class-prior changes.

For SVHN dataset, '0', '6', '8', '9' made up the P class, and '1', '2', '3', '4', '5', '7' made up the N class.

The model was AllConvNet BID45 as follows.0th (input) layer: (32*32*3)-1st to 3rd layers: [C(3*3, 96)]*2-C(3*3, 96, 2)- where C(3*3, 96) means 96 channels of 3*3 convolutions followed by ReLU, [ ?? ]*2 means 2 such layers, C(3*3, 96, 2) means a similar layer but with stride 2, etc.

Again, batch normalization and 2 -regularization with a regularization parameter 1e-5 were applied.

The optimizer was Adam with the default momentum parameters (?? 1 = 0.9 and ?? 2 = 0.999), an initial learning rate 1e-3, and a batch size 500.

This dataset consists of 60,000 32*32 color images in 10 classes, and there are 5,000 training images and 1,000 test images per class.

For CIFAR-10 dataset,??? the P class is composed of 'bird', 'cat', 'deer', 'dog', 'frog' and 'horse'; ??? the N class is composed of 'airplane', 'automobile', 'ship' and 'truck'. , 2016) .

The optimization setup was the same as for SVHN, except that the regularization parameter was set to be 5e-3 and the initial learning rate was set to be 1e-5.

In the experiments on the closeness of ?? and ?? and on the robustness against inaccurate training class priors, we sampled 40,000 training data from all the training data of MNIST in order to make it feasible to perform class-prior changes.

Final classification errors Please find in TAB3 .

We have compared the sigmoid loss sig (z) and the logistic loss log (z) on MNIST.

The experimental results are reported in Figure 4 .

We can see that the resulted classification errors are similar-in fact, sig (z) is a little better.

On the variation of n and n We have further investigated the issue of covariate shift by varying n and n .

Likewise, we test UU and CCN on MNIST by fixing n to 20,000 and gradually moving n from 20,000 to 4,000, where ?? is fixed to 0.4 and ?? is chosen from 0.9 or 0.8.

The experimental results in Figure 5 indicate that when n moves farther from n , UU and CCN become worse, while UU is affected slightly and CCN is affected severely.

Figure 5 is consistent with FIG3 , showing that CCN methods do not fit our problem setting.

<|TLDR|>

@highlight

Three class priors are all you need to train deep models from only U data, while any two should not be enough.

@highlight

Proposes an unbiased estimator that allows for training models with weak supervision on two unlabeled datasets with known class priors and discusses theoretical properties of the estimators.

@highlight

A methodology for training any binary classifier from only unlabeled data, and an empirical risk minimization method for two sets of unlabeled data where class priors are given.