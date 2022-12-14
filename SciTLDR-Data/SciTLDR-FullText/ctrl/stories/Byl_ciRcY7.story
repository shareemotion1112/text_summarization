A belief persists long in machine learning that enlargement of margins over training data accounts for the resistance of models to overfitting by increasing the robustness.

Yet Breiman shows a dilemma (Breiman, 1999) that a uniform improvement on margin distribution \emph{does not} necessarily reduces generalization error.

In this paper, we revisit Breiman's dilemma in deep neural networks with recently proposed normalized margins using Lipschitz constant bound by spectral norm products.

With both simplified theory and extensive experiments, Breiman's dilemma is shown to rely on dynamics of normalized margin distributions, that reflects the trade-off between model expression power and data complexity.

When the complexity of data is comparable to the model expression power in the sense that training and test data share similar phase transitions in normalized margin dynamics, two efficient ways are derived via classic margin-based generalization bounds to successfully predict the trend of generalization error.

On the other hand, over-expressed models that exhibit uniform improvements on training normalized margins may lose such a prediction power and fail to prevent the overfitting.

Margin, as a measurement of the robustness allowing some perturbations on classifier without changing its decision on training data, has a long history in characterizing the performance of classification algorithms in machine learning.

As early as BID17 , it played a central role in the proof on finite-stopping or convergence of perceptron algorithm when training data is separable.

Equipped with convex optimization technique, a plethora of large margin classifiers are triggered by support vector machines BID3 BID23 .

AdaBoost, an iterative algorithm to combine an ensemble of classifiers proposed by BID4 , often exhibits a resistance to overfitting phenomenon that during the training process the generalization error keeps on non-increasing when the training error drops to zero.

Toward deciphering the such a resistance of overfitting phenomenon, BID19 proposed an explanation that the training process keeps on improving a notion of classification margins in boosting, among later works on consistency of boosting with early stopping regularization BID2 BID30 BID28 .

Lately such a resistance to overfitting is again observed in deep neural networks with overparameterized models .

A renaissance of margin theory is proposed by BID0 with a normalization of network using Lipschitz constants bounded by products of operator spectral norms.

It inspires many further investigations in various settings BID14 BID16 BID12 .However, the improvement of margin distributions does not necessarily guarantee a better generalization performance, which is at least traced back to BID1 in his effort to understanding AdaBoost.

In this work, Breiman designed an algorithm arc-gv such that the margin can be maximized via a prediction game, then he demonstrated an example that one can achieve uniformly larger margin distributions on training data than AdaBoost but suffer a higher generalization error.

In the end of this paper, Breiman made the following comments with a dilemma: "The results above leave us in a quandary.

The laboratory results for various arcing algorithms are excellent, but the theory is in disarray.

The evidence is that if we try too hard to make the margins larger, then overfitting sets in. ...

My sense of it is that we just do not understand enough about what is going on."Breiman's dilemma triggers some further explorations to understand the limitation of margin theory in boosting BID18 Wang et al., 2008; BID27 .

In particular, BID18 points out that the trees found by arg-gv have larger model complexity in terms of deeper average depth than AdaBoost, suggesting that margin maximization in arc-gv does not necessarily control the model complexity.

The latter works provide tighter bounds based on VC-dimension and optimized quantile training margins, which however do not apply to over-parametrized models in deep neural networks and the case where the training margin distributions are uniformly improved.

In this paper, we are going to revisit Breiman's dilemma in the scenario of deep neural networks.

Both the success and failure can be seen on normalized margin based bounds on generalization error.

First of all, let's look at the following illustration example.

Example (Breiman's Dilemma with a CNN).

A basic 5-layer convolutional neural network of c channels (see Section 3 for details) is trained with CIFAR-10 dataset whose 10 percent labels are randomly permuted.

When c = 50 with 92, 610 parameters, FIG0 shows the training error and generalization (test) error in solid curves.

From the generalization error in (a) one can see that overfitting indeed happens after about 10 epochs, despite that training error continuously drops down to zero.

One can successfully predict such an overfitting phenomenon from FIG0 (b), the evolution of normalized margin distributions defined later in this paper.

In (b), while small margins are monotonically improved during training, large margins undergoes a phase transition from increase to decrease around 10 epochs such that one can predict the tendency of generalization error in (a) using large margin dynamics.

Two particular sections of large margin dynamics are highlighted in (b), one at 8.3 on x-axis that measures the percentage of normalized training margins no more than 8.3 (training margin error) and the other at 0.8 on y-axis that measures the normalized margins at quantile q = 0.8 (i.e. 1/?? q,t ).

Both of them meet the tendency of generalization error in (a) and find good early stopping time to avoid overfitting.

However, as we increase the channel number to c = 400 with about 5.8M parameters and retrain the model, (c) shows a similar overfitting phenomenon in generalization error; on the other hand, (d) exhibits a monotonic improvement of normalized margin distributions without a phase transition during the training and thus fails to capture the overfitting.

This demonstrates the Breiman's dilemma in CNN.

A key insight behind this dilemma, is that one needs a trade-off between the model expression power and the complexity of the dataset to endorse margin bounds a prediction power.

On one hand, when the model has a limited expression power relative to the training dataset, in the sense that the training margin distributions CAN NOT be uniformly improved during training, the generalization or test error may be predicted from dynamics of normalized margin distributions.

On the other hand, if we push too hard to improve the margin by giving model too much degree of freedom such that the training margins are uniformly improved during training process, the predictability may be lost.

A trade-off is thus necessary to balance the complexity of model and dataset, otherwise one is doomed to meet Breiman's dilemma when the models arbitrarily increase the expression power.

The example above shows that the expression power of models relative to the complexity of dataset, can be observed from the dynamics of normalized margins in training, instead of counting the number of parameters in neural networks.

In the sequel, our main contributions are to make these precise by revisiting the Rademacher complexity bounds with Lipschitz constants BID0 .???

With the Lipschitz-normalized margins, a linear inequality is established between training margin and test margin in Theorem 1.

When both training and test normalized margin distributions undergo similar phase transitions on increase-decrease during the training process, one may predict the generalization error based on the training margins as illustrated in FIG0 .???

In a dual direction, one can define a quantile margin via the inverse of margin distribution functions, to establish another linear inequality between the inverse quantile margins and the test margins as shown in Theorem 2.

Quantile margin is far easier to tune in practice and enjoys a stronger prediction power exploiting an adaptive selection of margins along model training.??? In all cases, Breiman's dilemma may fail both of the methods above when dynamics of normalized training margins undergo different phase transitions to that of test margins during training, where a uniform improvement of margins results in overfitting.

Section 2 describes our method to derive the two linear inequalities of generalization bounds above.

Extensive experimental results are shown in Section 3 and Appendix with basic CNNs, AlexNet, VGG, ResNet, and various datasets including CIFAR10, CIFAR100, and mini-Imagenet.

Let X be the input space (e.g. X ??? R C??W ??H in image classification) and Y := {1, . . .

, K} be the space of K classes.

Consider a sample set of n observations S = {(x 1 , y 1 ), . . .

, (x n , y n ) : x i ??? X , y i ??? Y} that are drawn i.i.d.

from P X,Y .

For any function f : X ??? R, let Pf = X f (X)dP be the population expectation and P n f = (1/n) n i=1 f (x i ) be the sample average.

Define F to be the space of functions represented by neural networks, DISPLAYFORM0 where l is the depth of the network, W i is the weight matrix corresponding to a linear operator on x i and ?? i stands for either element-wise activation function (e.g. ReLU) or pooling operator that are assumed to be Lipschitz bounded with constant L ??i and satisfying ?? i (0) = 0.

For example, in convolutional network, W i x i + b i = w i * x i + b i where * stands for the convolution between input tensor x l and kernel tensor w l .

We equip F with the Lipschitz semi-norm, for each f , DISPLAYFORM1 where ?? ?? is the spectral norm and DISPLAYFORM2 For all the examples in this paper, we use ReLU activation ?? i that leads to L ??i = 1.

Moreover we consider the following family of hypothesis mapping, DISPLAYFORM3 where [??] j denotes the j th coordinate and we further define the following class induced by Lipschitz semi-norm bound on F, DISPLAYFORM4 Lastly, rather than merely looking at whether a prediction f (x) on y is correct or not, we also consider the margin defined as ??(f (x), y) = [f (x)]

y ???max {j:j =y} [f (x)] j .

Therefore, we can define the ramp loss and margin error depending on the confidence of predictions.

Given two thresholds ?? 2 > ?? 1 ??? 0, define a ramp loss to be DISPLAYFORM5 where ??? := ?? 2 ??? ?? 1 .

In particular ?? 1 = 0 and ?? 2 = ??, we also write ?? = ?? for simplicity.

Define the margin error to measure if f has margin no more than a threshold ??, DISPLAYFORM6 In particular, e 0 (f (x), y) is the common mis-classification error and DISPLAYFORM7 Note that e 0 ??? ?? ??? e ?? , and ?? is Lipschitz bounded by 1/??.

The central question we try to answer is, can we find a proper upper bound to predict the tendency of the generalization error along training, such that one can early stop the training near the epoch that DISPLAYFORM8 The answer is both a yes and a no!We begin with the following lemma, as a typical result in multi-label classification from the uniform law of large numbers BID8 .

Lemma 2.1.

Given a ?? 0 > 0, then, for any ?? ??? (0, 1), with probability at least 1 ??? ??, the following holds for any f ??? F with f F ??? L, DISPLAYFORM9 is the Rademacher complexity of function class H L with respect to n samples, and the expectation is taken over x i , ?? i , i = 1, ..., n.

Unfortunately, direct application of such bound for a constant ?? 0 will suffer from the so-called scaling problem.

The following proposition gives an lower bound of Rademacher complexity term, whose proof is provided in Appendix D. Proposition 1.

Consider the networks with ReLU activation functions.

For any L > 0, there holds, DISPLAYFORM10 where C > 0 is a constant that does not depend on S.The lemma tells us if L ??? ???, upper bound (6) becomes trivial since R n (H L ) ??? ???. In fact, both BID22 and BID21 show that with gradient descent, the norm of estimator's weight in logistic regression and general boosting (including exponential loss), respectively, will go to infinity at a growth rate log(t) when the data is linearly separable.

As for the deep neural network with cross-entropy loss, the input of last layer is usually be viewed as features extracted from original input.

Training the last layer with other layers fixed is exactly a logistic regression, and the feature is linearly separable as long as the training error achieves zero.

Therefore, without any normalization, the hypothesis space along training has no upper bound on L and the upper bound (6) is useless.

Besides, even for a fixed L, the complexity term R n (H L ) is computationally intractable.

The first remedy is to restrict our attention on H 1 by normalizing f with its Lipschitz semi-norm f F or its upper bounds.

Note that a normalized networkf = f /C has the same mis-classification error as f for all C > 0.

For the choice of C, it's hard in practice to directly compute the Lipschitz semi-norm of a network, but instead some approximate estimates on the upper bound L f in (2) are available as discussed in Appendix A. In the sequel, letf = f /L f be the normalized network and DISPLAYFORM11 be the corresponding normalized hypothesis function.

Now a simple idea is to regard R n (H 1 ) as a constant and predict the tendency of generalization error via training margin error of the normalized network, that avoids the scaling problem and the computation of complexity term.

The following theorem makes this precise.

Theorem 1.

Given ?? 1 and ?? 2 such that ?? 2 > ?? 1 ??? 0 and ??? := ?? 2 ??? ?? 1 ??? 0, for any ?? > 0, with probability at least 1 ??? ??, along the training epoch t = 1, . . .

, T , the following holds for each f t , DISPLAYFORM12 where DISPLAYFORM13 Remark.

In particular, when we take ?? 1 = 0 and ?? 2 = ?? > 0, the bound above becomes, DISPLAYFORM14 Theorem 1 says, we can bound the normalized test margin distribution DISPLAYFORM15 Recently BID12 investigates for normalized networks, the strong linear relationship between cross-entropy training loss and test loss when the training epochs are large enough.

As a contrast, we consider the whole training process and normalized margins.

In particular, we hope to predict the trend of generalization error by choosing ?? 1 = 0 and a proper ??.

For this purpose, the following facts are important.

First, we do not expect the bound, for example (10), is tight for every choice of ?? > 0, instead we hope there exists some ?? such that the training margin error nearly monotonically changes with generalization error.

FIG1 shows the existence of such ?? such that the training margin error successfully recover the tendency of generalization error on CIFAR10 dataset.

Moreover, in Appendix Figure 8 shows the rank correlation between training margin error at various ?? and training/test error.

Second, the normalizing factor is not necessarily to be an upper bound of Lipschitz semi-norm.

The key point is to prevent the complexity term of the normalized network going to infinity.

Since for any constant c > 0, normalization byL = cL works in practice where the constant could be absorbed to ??, we could ignore the Lipschitz constant introduced by general activation functions in the middle layers.

However, it is a natural question whether a reasonable ?? with prediction power exists.

A simple example in FIG0 shows, once the training margin distribution is uniformly improved, dynamic of training margin error fails to detect the minimum of generalization error in the early stage.

This is because when network structure becomes complex enough, the training margin distribution could be more easily improved but the the generalization error may overfit.

This is exactly the same observation in BID1 to doubt the margin theory in boosting type algorithms.

More detailed discussions will be given in Section 3.2.The most serious limitation of Theorem 1 lies in we must fix a ?? along the complete training process.

In fact, the first term and second term in the bound (10) vary in the opposite directions with respect to ??, and thus different f t may prefer different ?? for a trade-off.

As in FIG0 (b) of the example, while choosing ?? is to fix an x-coordinate section of margin distributions, its dual is to look for a y-section which leads to different margins for different f t .

This motivates the quantile margin in the following theorem.

Let?? q,f be the q th quantile margin of the network f with respect to sample S, DISPLAYFORM16 Theorem 2.

Assume the input space is bounded by M > 0, that is x 2 ??? M, ???x ??? X .

Given a quantile q ??? [0, 1], for any ?? ??? (0, 1) and ?? > 0, the following holds with probability at least 1 ??? ?? for all f t satisfying?? q,ft > ?? , DISPLAYFORM17 DISPLAYFORM18 Remark.

We simply denote ?? q,t for ?? q,ft when there is no confusion.

Compared with the bound (10), (12) make the choice of ?? varying with f t and the cost is an additional constant term C 2 q and the constraint?? q,t > ?? that typically holds for large enough q in practice.

In applications, stochastic gradient descent (SGD) often effectively improves the training margin distributions along the drops of training errors, a small enough ?? and large enough q usually meet?? q,t > ?? .

Moreover, even with the choice ?? = exp(???B), constant term [log log 2 (4(M + l)/?? )]/n = O( log B/n) is still negligible and thus very little cost is paid in the upper bound.

In practice, tuning q ??? [0, 1] is far easier than tuning ?? > 0 directly and setting a large enough q ??? 0.9 usually provides us lots of information about the generalization performance.

The quantile margin works effectively when the dynamics of large margin distributions reflects the behavior of generalization error, e.g. FIG0 .

In this case, after certain epochs of training, the large margins have to be sacrificed to further improve small margins to reduce the training loss, that typically indicates a possible saturation or overfitting in test error.

We briefly introduce the network and dataset used in the experiments.

For the network, we first consider the convolutional neural network with very simple structure basic CNN(c).

The structure is shown in Appendix Figure 7 .

Basically, it has five convolutional layers with c channels at each and one fully connected layer, where c will be specified in concrete examples.

Second, we consider more practical network structure, AlexNet BID10 , VGGNet-16 BID20 and ResNet-18 BID6 .

For the dataset, we consider CIFAR10, CIFAR100 BID9 ) and Mini-ImageNet .The spirit of the following experiments is to show, when and how, the margin bound could be used to predict the tendency of generalization or test error along the training path?

This section is to apply Theorem 1 and Theorem 2 to predict the tendency of generalization error.

Let's firstly consider training a basic CNN(50) on CIFAR10 dataset with and without random noise.

The relations between generalization error and training margin error e ?? (f (x), y) with ?? = 9.8, inverse quantile margin 1/?? q,t with q = 0.6 are shown in FIG1 .

In this simple example where the net is light and the dataset is simple, the linear bounds (9) and (12) show a good prediction power: they stop either near the epoch of sufficient training (Left, original data) or where even an overfitting occurs (Right, 10 percents label corrupted).

and CIFAR10 with 10 percents label corrupted (Right).

In each figure, we show training error (red solid), training margin error ?? = 10 (red dash) and inverse quantile margin (red dotted) with q = 0.6 and generalization error (blue solid).

The marker "x" in each curve indicates the global minimum along epoch 1, . . .

, T .

Both training margin error and inverse quantile margin successfully predict the tendency of generalization error.

A few discussions are given below.1.

There exists a trade-off on the choice of ?? from the linear bounds (9) (and parallel arguments hold for q).

The training margin error with a small ?? is close to the training error, while a large ?? is close to generalization error and it's illustrated in Appendix Figure 8 where we show the Spearman's ?? rank correlation 1 between training margin error and training error, generalization error against threshold ??.

2.

The training margin error (or inverse quantile margin) is closely related to the dynamics of training margin distributions.

For certain choice of ??, if the curve of training margin error (with respect to epoch) is V-shape, the corresponding dynamics of training margin distributions will have a cross-over, where the low margins have a monotonic increase and the large margins undergo a phase transition from increase to decrease, as illustrated by the red arrow in FIG0 .

3.

Dynamics of quantile margins can adaptively select ?? t for each f t without access to the complexity term.

Unlike merely looking at the training margin error with a fixed ??, quantile margin bound (12) shows a stronger prediction power than (10) and even be able to capture more local information as illustrated in FIG2 .

The generalization error curve has two valleys corresponding to a local optimum and a global optimum, and the quantile margin curve with q = 0.95 successfully identifies both.

However, if we consider the dynamics of training margin errors, it's rarely possible to recover the two valleys at the same time since their critical thresholds ?? t1 and ?? t2 are different.

Another example of ResNet is given in Appendix Figure 9 .

In this section, we explore the normalized margin dynamics with over-parameterized models whose expression power might be greater than data complexity.

We conduct experiments in the following two scenarios.1.

In the first experiment shown in FIG3 , we fix the dataset to be CIFAR10 with 10 percent of labels randomly permuted, and gradually increase the channels from basic CNN(50) to basic CNN(400).

As the channel number increases, dynamics of the normalized training margins in the first row change from a phase transition with a cross-over in large margins to a monotone improvement of margin distributions.

This phenomenon is not a surprise since with a strong representation power, the whole training margin distribution can be monotonically improved without sacrificing the large margins.

On the other hand, the generalization or test error can never be monotonically improved.

In the second row, heatmaps depict rank correlations of dynamics between training and test margin errors, which clearly show the phase transitions for CNN(50) and CNN(100) and its disappearance for CNN(400).

2.

In the second experiment shown in 5, we compare the normalized margin dynamics of training CNN(400) and ResNet18 on two different datasets, CIFAR100 (the simpler) and Mini-ImageNet (the more complex).

It shows that: (a) CNN(400) (5.8M parameters) does not have an over-representation power on CIFAR100, whose normalized training margin dynamics exhibits a phase transition -a sacrifice of large margins to improve small margins during training; (b) ResNet18 (11M parameters) exhibits an over-representation power on CIFAR100 via a monotone improvement on training margins, but loses such a power in Mini-ImageNet with the phase transitions in margin dynamics.

More experiments including AlexNet and VGG16 are shown in Appendix FIG0 .This phenomenon is not unfamiliar to us, since Breiman BID1 has pointed out that the improvement of training margins is not enough to guarantee a small generalization or test error in the boosting type algorithms.

In this paper Breiman designed an algorithm, called arc-gv, enjoying an uniformly better training margin distribution comparing with Adaboost but suffer a higher generalization error.

Now again we find the same phenomenon ubiquitous in deep neural networks.

Dataset: CIFAR100 (Left, Middle), Mini-ImageNet (Right) with 10 percent labels corrupted.

With a fixed network structure, we further explore how the complexity of dataset influences the margin dynamics.

Taking ResNet18 as an example, margin dynamics on CIFAR100 doesn't have any crossover (phase transition), but on Mini-Imagenet a cross-over occurs.

In the end, it's worth mentioning different choices of the normalization factor estimates may affect the range of predictability.

In all experiments above, normalization factor is estimated via an upper bound on spectral norm given in Appendix A (Lemma A.1 in Section A).

One could also use power iteration BID14 to present a more precise estimation on spectral norm.

It turns out a more accurate estimation of spectral norm can extend the range of predictability, but Breiman's dilemma is still there when the balance between model expression power and dataset complexity is broken.

More experiments on this aspect can be found in FIG0 in Appendix.

In this paper, we show that Breiman's dilemma is ubiquitous in deep learning, in addition to previous studies on Boosting algorithms.

We exhibit that Breiman's dilemma is closely related to the tradeoff between model expression power and data complexity.

A novel perspective on phase transitions in dynamics of Lipschitz-normalized margin distributions is proposed to inspect when the model has over-representation power compared to the dataset, instead of merely counting the number of parameters.

A data-driven early stopping rule by monitoring the margin dynamics is a future direction to explore.

Lipschitz semi-norm plays an important role in normalizing or regularizing neural networks, e.g. in GANs BID7 BID14 , therefore a more careful treatment deserves further pursuits.

In this section we discuss how to estimate the Lipschitz constant bound in (2).

Given an operator W associated with a convolutional kernel w, i.e. W x = w * x, there are two ways to estimate its operator norm.

We begin with a useful lemma, Lemma A.1.

For convolution operator with kernel w, i.e. W x := w * x, there holds w * x 2 ??? w 1 x 2 .In other words, W ?? ??? w 1 .Proof.

DISPLAYFORM0 where the second last step is due to Cauchy-Schwartz inequality.

A. 1 -norm.

The convolutional operator (spectral) norm can be upper bounded by the 1 -norm of its kernels, i.e. W ?? ??? w 1 .

This is a simple way but the bound gets loose when the channel numbers increase.

B. Power iteration.

A fast approximation for the spectral norm of the operator matrix is given in BID14 in GANs that is based on power iterations BID5 ).

Yet as a shortcoming, it is not easy to apply to the ResNets.

We compare two estimation in Appendix FIG0 .

It turns out both of them have prediction power on the tendency of generalization error and both of them will fail when the network has large enough expression power.

Though using 1 norm of kernel is extremely efficient, the power iteration method may be tighter and has a wider range of predictability.

In the remaining of this section, we will particularly discuss the treatment of ResNets.

ResNet is usually a composition of the basic blocks shown in FIG5 with short-cut structure.

The following method is used in this paper to estimate the upper bound of operator or spectral norm of such a basic block of ResNet.

B are mean and variance of batch samples, while keeping an online averaging as?? and?? 2 .

Then BN rescales x + by estimated parameters??,?? and outputx =??x + +??.

Therefore the whole rescaling of BN on the kernel tensor w of the convolution layer is?? = w??/ ????? 2 + and its corresponding rescaled operator is DISPLAYFORM1 (b) Activation and pooling: their Lipschitz constants L ?? can be known a priori, e.g. L ?? = 1 for ReLU and hence can be ignored.

In general, L ?? can not be ignored if they are in the shortcut as discussed below.(d) Shortcut: In residue net with basic block in FIG5 , one has to treat the mainstream (Block 2 , Block 3 ) and the shortcut Block 1 separately.

Since f + g F ??? f F + g F , in this paper we take the Lipschitz upper bound by DISPLAYFORM2 where ?? i ?? denotes a spectral norm estimate of BN-rescaled convolutional operator W i .

In particular L ??out can be ignored since all paths are normalized by the same constant while L ??in can not be ignored due to its asymmetry.

B STRUCTURE OF BASIC CNN The picture is slight different here, since after the first (better) local minimum, the training margin distribution is uniformly improved without reducing generalization error.

Therefore, we could not expect the inverse quantile margin to reflect the tendency of generalization error globally, especially the order of two local minimums.

However, around epochs when local minimum occurs, the training margin distribution still has a cross-over, and thus the inverse quantile margin could reflect the tendency locally.

Lemma D.1.

For any ?? ??? (0, 1) and bounded-value functions F B := {f : X ??? R : f ??? ??? B}, the following holds with probability at least 1 ??? ??, DISPLAYFORM3 where DISPLAYFORM4 is the Rademacher Complexity of function class F.For completeness, we include its proof that also needs the following well-known McDiarmid's inequality (see, e.g. Wainwright (2019) ).

DISPLAYFORM5 where with probability at least 1 ??? ??, DISPLAYFORM6 by McDiarmid's bounded difference inequality, and DISPLAYFORM7 using Rademacher complexity.

To see FORMULA0 , we are going to show that sup f ???F B E nf is a bounded difference function.

Consider DISPLAYFORM8 Assume that the i-th argument x i changes to x i , then for every g, g( DISPLAYFORM9 Hence sup g g(x i , x ???i ) ??? sup g g(x i , x ???i ) ??? B/n, which implies that sup f ???F B E nf is a B/nbounded difference function.

Then (16) follows from the McDiarmid's inequality (Lemma D.2) using B i = B/n and ?? = exp(???2n?? 2 /B 2 ).As to (17), DISPLAYFORM10 that ends the proof.

We also need the following contraction inequality of Rademacher Complexity BID11 BID13 .

Lemma D.3 (Rademacher Contraction Inequality).

For any Lipschitz function: DISPLAYFORM11 Ledoux & Talagrand (1991) has an additional factor 2 in the contraction inequality which is dropped in BID13 .

Its current form is stated in BID15 as Talagrand's Lemma (Lemma 4.2).Beyond, we further introduce the family, DISPLAYFORM12 and the sub-family constraint in Lipschitz semi-norm on f , DISPLAYFORM13 The following lemma BID8 allows us to bound the Rademacher complexity term of R n (G) by R n (H), DISPLAYFORM14 where the last inequality is implied from R n ({max(f 1 , . . .

, f M ) : BID8 BID15 .

DISPLAYFORM15

Proof of Proposition 1.

Without loss of generality, we assume L ??i = 1, i = 1, . . .

, l. Let T (r) =: {t(x) = w ?? x : w 2 ??? r} be the class of linear function with Lipschitz semi-norm less than r and we show that for each t ??? T (L/2), there exists f ??? F with f F ??? L and y 0 ??? {1, . . .

, K} such that where f F ??? ?? l i=1 W i ?? = 2L/2 = L, and thus h ??? H L by definition.

Therefore, R n (H L ) ??? R n (T (L/2)), DISPLAYFORM0 DISPLAYFORM1 where the second equality is implied from Cauchy-Schwarz inequality and the last inequality is implied from Khintchine inequality.

Proof of Theorem 1.

Consider l (??1,??2) (??(f (x), y)), wheref := f /L f is the normalized network, ??(f (x), y) ??? G 1 .

Then for any ?? 2 > ?? 1 ??? 0, DISPLAYFORM0 ??? P n (??1,??2) (f (x), y) + 2R n (l (??1,??2) ??? G 1 ) + log(1/??) 2n , ??? P n (??1,??2) (f (x), y) + 2 ??? R n (G 1 ) + log(1/??) 2n , ??? P n ??1,??2 (f (x), y) + 2K 2 ??? R n (H 1 ) + log(1/??) 2n , ??? P n ??2 (f (x), y) + 2K 2 ??? R n (H 1 ) + log(1/??) 2n , where the first and last inequality is implied from 1[?? < ?? 1 ] ??? (??1,??2) (??) ??? 1[?? < ?? 2 ], the second inequality is a direct consequence of Lemma D.1, the third inequality results from Rademacher Contraction Inequality (Lemma D.3) and finally the fourth equation is implied from Lemma D.4.

Proof of Theorem 2.

Firstly, we show after normalization, the normalize margin has an upper bound, DISPLAYFORM0 , where x i = ?? i (W i x i???1 + b i ) with x 0 = x,W i = (W i , b i ) and L ??i is the Lipschitz constant of activation function ?? i with ?? i (0) = 0, i = 1, . . .

, L. Then, for normalized networkf = f /L f with DISPLAYFORM1 Therefore ??(f (x), y) ??? 2 f (x) 2 = 2(M + L) =: M 1 , and the quantile margin is also bounded ?? q,t ??? M 1 for all q ??? (0, 1), t = 1, . . .

, T .The remaining proof is standard.

For any > 0, we take a sequence of k and ?? k , k = 1, 2, . . .

by k = + log k n and ?? k = M 1 2 ???k .

Then by Theorem 1, DISPLAYFORM2 where A k is the event P[??(f t (x), y) < 0] > P n [??(f (x), y) < ?? k ] + 2K 2 ?? k R(H 1 ) + k , and the probability is taken over samples {x 1 , ...x n }.

We further consider the probability for none of A k occurs, DISPLAYFORM3 2 ), ??? 2 exp(???2n 2 ).Hence, fix a q ??? [0, 1], for any t = 1, . . .

, T , as long as?? q,t > 0, there exists ak ??? 1 such that, ??k +1 ????? q,t < ??k.

Therefore, DISPLAYFORM4 ??? P[??(f t (x), y) < 0] > P n [??(f t (x), y) <?? q,t ] + 4K 2 ?? q,t R(H 1 ) + k +1 , = P[??(f t (x), y) < 0] > P n [??(f t (x), y) >?? q,t ] + 4K 2 ?? q,t R(H 1 ) + + log(k + 1) n , ??? P[??(f t (x), y) < 0] > P n [??(f t (x), y) >?? q,t ] + 4K 2 ?? q,t R(H 1 ) + + log log 2 (2M 1 /?? q,t ) n .The first inequality is implied from P n [??(f t (x), y) <?? q,t ] > P n [??(f t (x), y) < ??k +1 ], since ??k +1 ??? ?? q,t .

The second inequality is implied from?? q,t < 2??k +1 and thus, 1/??k +1 < 2/?? q,t .

The third equality is the direct definition of k .

The last inequality is implied fromk + 1 = log 2 (M 1 /??k +1 ) and again, 1/??k +1 < 2/?? q,t .

The conclusion is proved immediately if we do a transform from to ??.

<|TLDR|>

@highlight

Bregman's dilemma is shown in deep learning that improvement of margins of over-parameterized models may result in overfitting, and dynamics of normalized margin distributions are proposed to predict generalization error and identify such a dilemma. 