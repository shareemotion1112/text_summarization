This paper is focused on investigating and demystifying an intriguing robustness phenomena in over-parameterized neural network training.

In particular we provide empirical and theoretical evidence that first order methods such as gradient descent are provably robust to noise/corruption on a constant fraction of the labels despite over-parameterization under a rich dataset model.

In particular: i) First, we show that in the first few iterations where the updates are still in the vicinity of the initialization these algorithms only fit to the correct labels essentially ignoring the noisy labels.

ii) Secondly, we prove that to start to overfit to the noisy labels these algorithms must stray rather far from from the initial model which can only occur after many more iterations.

Together, these show that gradient descent with early stopping is provably robust to label noise and shed light on empirical robustness of deep networks as well as commonly adopted early-stopping heuristics.

1. Introduction

This paper focuses on an intriguing phenomena: overparameterized neural networks are surprisingly robust to label noise when first order methods with early stopping is used to train them.

To observe this phenomena consider FIG1 where we perform experiments on the MNIST data set.

Here, we corrupt a fraction of the labels of the training data by assigning their label uniformly at random.

We then fit a four layer model via stochastic gradient descent and plot various performance metrics in Figures 1a and 1b.

FIG1 (blue curve) shows that indeed with a sufficiently large number of iterations the neural network does in fact perfectly fit the corrupted training data.

However, FIG1 also shows that such a model does not generalize to the test data (yellow curve) and the accuracy with respect to the ground truth labels degrades (orange curve).

These plots clearly demonstrate that the model overfits with many iterations.

In FIG1 we repeat the same experiment but this time stop the updates after a few iterations (i.e. use early stopping).

In this case the train accuracy degrades linearly (blue curve).

However, perhaps unexpected, the test accuracy (yellow curve) remains high even with a significant amount of corruption.

This suggests that with early stopping the model does not overfit and generalizes to new test data.

Even more surprising, the train accuracy (orange curve) with respect to the ground truth labels continues to stay around %100 even when %50 of the labels are corrupted.

That is, with early stopping overparameterized neural networks even correct the corrupted labels!

These plots collectively demonstrate that overparameterized neural networks when combined with early stopping have unique generalization and robustness capabilities.

As we detail further in Section D this phenomena holds (albeit less pronounced) for richer data models and architectures.

This paper aims to demonstrate and begin to demystify the surprising robustness of overparameterized neural networks when early stopping is used.

We show that gradient descent is indeed provably robust to noise/corruption on a constant fraction of the labels in such overparametrized learning scenarios.

In particular, under a fairly expressive dataset model and focusing on one-hidden layer networks, we show that after a few iterations (a.k.a.

early stopping), gradient descent finds a model (i) that is within a small neighborhood of the point of initialization and (ii) only fits to the correct labels essentially ignoring the noisy labels.

We complement these findings by proving that if the network is trained to overfit to the noisy labels, then the solution found by gradient descent must stray rather far from the initial model.

Together, these results highlight the key features of a solution that generalizes well vs a solution that fits well.

We now describe the dataset model used in our theoretical results.

In this model we assume that the input samples x 1 , x 2 , . . .

, x n ??? R d come from K clusters which are located on the unit Euclidian ball in R d .

We also assume our data set consists ofK ??? K classes where each class can be composed of multiple clusters.

We consider a deterministic data set with n samples with roughly balanced clusters each In these experiments we use a 4 layer neural network consisting of two convolution layers followed by two fully-connected layers to train a data set of 50,000 samples from MNIST with various amounts of random corruption on the labels.

In this architecture the convolutional layers have width 64 and 128 kernels, and the fully-connected layers have 256 and 10 outputs, respectively.

Overall, there are 4.8 million trainable parameters.

We depict the training accuracy both w.r.t.

the corrupted and uncorrupted training labels as well as the (uncorrupted) test accuracy.

(a) Shows the performance after 200 epochs of Adadelta where near perfect fitting to the corrupted data is achieved.

(b) Shows the performance with early stopping.

We observe that with early stopping the trained neural network is robust to label corruption.consisting on the order of n K samples.1 Finally, while we allow for multiple classes, in our theoretical model we assume the labels are scalars and take values in [???1, 1] interval.

We formally define our dataset model below and provide an illustration in Figure 2 .

Definition 1.1 (Clusterable dataset) Consider a data set of size n consisting of input/label pairs 1 This is for ease of exposition rather than a particular challenge arising in the analysis.

DISPLAYFORM0 We assume the input data have unit Euclidean norm and originate from K clusters with the th cluster containing n data points.

We assume the number of points originating from each cluster is well-balanced in the sense that c low n K ??? n ??? c up n K with c low and c up two numerical constants obeying 0 < c low < c up < 1.

We use {c } K =1 ??? R d to denote the cluster centers which are distinct unit Euclidian norm vectors.

We assume the input data points x that belong to the -th cluster obey x ??? c 2 ??? ?? 0 , with ?? 0 > 0 denoting the input noise level.

We assume the labels y i belong to one ofK ??? K classes.

Specifically, we assume y i ??? {?? 1 , ?? 2 , . . .

, ??K} with {?? }K =1 ??? [???1, 1] denoting the labels associated with each class.

We assume all the elements of the same cluster belong to the same class and hence have the same label.

However, a class can contain multiple clusters.

Finally, we assume the labels are separated in the sense that ?? r ??? ?? s ??? ?? for r ??? s,(1.1) with ?? > 0 denoting the class separation.

In the data model above {c } K =1 are the K cluster centers that govern the input distribution.

We note that in this model different clusters can be assigned to the same label.

Hence, this setup is rich enough to model data which is not linearly separable: e.g. over R 2 , we can assign cluster centers (0, 1) and (0, ???1) to label 1 and cluster centers (1, 0) and (???1, 0) to label ???1.

Note that the maximum number of classes are dictated by the separation ??.

In particular, we can have at mostK ??? 2 ?? +1 classes.

We remark that this model is related to the setup of (4) which focuses on providing polynomial guarantees for learning shallow networks.

Finally, note that, we need some sort of separation between the cluster centers to distinguish them.

While Definition 1.1 doesn't specifies such separation explicitly, Definition 2.1 establishes a notion of separation in terms of how well a neural net can distinguish the cluster centers.

Next, we introduce our noisy/corrupted dataset model.

DISPLAYFORM1 be an (?? 0 , ??) clusterable dataset with ?? 1 , ?? 2 , . . . , ??K denoting theK possible class labels.

DISPLAYFORM2 as follows.

For each cluster 1 ??? ??? K, at most ??n of the labels associated with that cluster (which contains n points) is assigned to another label value chosen from {?? }K =1 .

We shall refer to the initial labels {??? i } n i=1 as the ground truth labels.

We note that this definition allows for a fraction ?? of corruptions in each cluster.

1 Figure 2 .

Visualization of the input/label samples and classes according to the clusterable dataset model in Definition 1.1.

In the depicted example there are K = 6 clusters,K = 3 classes.

In this example the number of data points is n = 30 with each cluster containing 5 data points.

The labels associated to classes 1, 2, and 3 are ??1 = ???1, ??2 = 0.1, and ??3 = 1, respectively so that ?? = 0.9.

We note that the placement of points are exaggerated for clarity.

In particular, per definition the cluster center and data points all have unit Euclidean norm.

Also, there is no explicit requirements that the cluster centers be separated.

The depicted separation is for exposition purposes only.

Network model: We will study the ability of neural networks to learn this corrupted dataset model.

To proceed, let us introduce our neural network model.

We consider a network with one hidden layer that maps R d to R. Denoting the number of hidden nodes by k, this network is characterized by an activation function ??, input weight matrix W ??? R k??d and output weight vector v ??? R k .

In this work, we will fix output v to be a unit vector where half the entries are 1 ??? k and other half are ???1 ??? k to simplify exposition.2 We will only optimize over the weight matrix W which contains most of the network parameters and will be shown to be sufficient for robust learning.

We will also assume ?? has bounded first and second order derivatives, i.e. ?? ??? (z) , ?? ?????? (z) ??? ?? for all z. The network's prediction at an input sample x is given by DISPLAYFORM3 where the activation function ?? applies entrywise.

Given DISPLAYFORM4 , we shall train the network via minimizing the empirical risk over the training data via a quadratic loss DISPLAYFORM5 In particular, we will run gradient descent with a constant learning rate ??, starting from a random initialization W 0 via the following updates DISPLAYFORM6 2 If the number of hidden units is odd we set one entry of v to zero.

Throughout, ??? denotes the largest singular value of a given matrix.

The notation O(???) denotes that a certain identity holds up to a fixed numerical constant.

Also, c, c 0 , C, C 0 etc.

represent numerical constants.

Our main result shows that overparameterized neural networks, when trained via gradient descent using early stopping are fairly robust to label noise.

The ability of neural networks to learn from the training data, even without label corruption, naturally depends on the diversity of the input training data.

Indeed, if two input data are nearly the same but have different uncorrupted labels reliable learning is difficult.

We will quantify this notion of diversity via a notion of condition number related to a covariance matrix involving the activation ?? and the cluster centers {c } K =1 .Definition 2.1 Define the matrix of cluster centers DISPLAYFORM0 Define the neural net covariance matrix ??(C) as DISPLAYFORM1 Here ??? denotes the elementwise product.

Also denote the minimum eigenvalue of ??(C) by ??(C) and define the condition number associated with the cluster centers C as DISPLAYFORM2 One can view ??(C) as an empirical kernel matrix associated with the network where the kernel is given by DISPLAYFORM3 Note that ??(C) is trivially rank deficient if there are two cluster centers that are identical.

In this sense, the minimum eigenvalue of ??(C) will quantify the ability of the neural network to distinguish between distinct cluster centers.

Therefore, one can think of ??(C) as a condition number associated with the neural network which characterizes the distinctness/diversity of the cluster centers.

The more distinct the cluster centers, the larger ??(C) and smaller the condition number ??(C) is.

Indeed, based on results in (5) when the cluster centers are maximally diverse e.g. uniformly at random from the unit sphere ??(C) scales like a constant.

Throughout we shall assume that ??(C) is strictly positive (and hence ??(C) < ???).

This property is empirically verified to hold in earlier works (6) when ?? is a standard activation (e.g. ReLU, softplus).

As a concrete example, for ReLU activation, using results from (5) one can show if the cluster centers are separated by a distance ?? > 0, then ??(C) ??? ?? 100K 2 .

We note that variations of the ??(C) > 0 assumption based on the data points (i.e. ??(X) > 0 not cluster centers) (5; 7; 8) are utilized to provide convergence guarantees for DNNs.

Also see (9; 10) for other publications using related definitions.

With a quantitative characterization of distinctiveness/diversity in place we are now ready to state our main result.

Throughout we use c ?? , C ?? , etc.

to denote constants only depending on ??. We note that this Theorem is slightly simplified by ignoring logarithmic terms and precise dependencies on ??. See Theorem E.13 for precise statements.

DISPLAYFORM4 with ??(C) the neural net cluster condition number pre Definition 2.1.

Then as long as 0 ???c ?? K 2 and ?? ??? ?? 8 with probability at least 1 ??? 3 K 100 , after DISPLAYFORM5 ) iterations, the neural network 3 If k is odd we set one entry to zero ??? DISPLAYFORM6 f (???, W ??0 ) found by gradient descent assigns all the input samples x i to the correct ground truth labels??? i .

That is, arg min DISPLAYFORM7 holds for all 1 ??? i ??? n. Furthermore, for all 0 ??? ?? ??? ?? 0 , the distance to the initial point obeys DISPLAYFORM8 Theorem 2.2 shows that gradient descent with early stopping has a few intriguing properties: Robustness.

The solution found by gradient descent with early stopping degrades gracefully as the label corruption level ?? grows.

In particular, as long as ?? ??? ?? 8, the final model is able to correctly classify all samples including the corrupted ones.

In our setup, intuitively label gap obeys ?? ??? 1 K , hence, we prove robustness to Total Number of corrupted labels ??? n K .This result is independent of number of clusters and only depends on number of classes.

An interesting future direction is to improve this result to allow on the order of n corrupted labels.

Such a result maybe possible by using a multi-output classification neural network.

Early stopping time.

We show that gradient descent finds a model that is robust to outliers after a few iterations.

In particular using the maximum allowed step size, the number of iterations is of the order of DISPLAYFORM9 ) which scales with K d up to condition numbers.

Modest overparameterization.

Our result requires modest overparemetrization and apply as soon as the number of parameters exceed the number of classes to the power four (kd ??? K 4 ).

Interestingly, the amount of overparameterization is essentially independent of the size of the training data n (ignoring logarithmic terms) and conditioning of the data points, only depending on the number of clusters and conditioning of the cluster centers.

This can be interpreted as ensuring that the network has enough capacity to fit the cluster centers {c } K =1 and the associated true labels.

Distance from initialization.

Another feature of Theorem 2.2 is that the network weights do not stray far from the initialization as the distance between the initial model and the final model (at most) grows with the square root of the number of clusters ( ??? K).

This ??? K dependence implies that the more clusters there are, the updates travel further away but continue to stay within a certain radius.

This dependence is intuitive as the Rademacher complexity of the function space is dictated by the distance to initialization and should grow with the square-root of the number of input clusters to ensure the model is expressive enough to learn the dataset.

We would like to note that in the limit of 0 ??? 0 where the input data set is perfectly clustered one can improve the amount of overparamterization.

Indeed, the result above is obtained via a perturbation argument from this more refined result stated below.

Theorem A.1 (Training with perfectly clustered data) Consier the setting and assumptions of Theorem E.14 with 0 = 0.

Starting from an initial weight matrix W 0 selected at random with i.i.d.

N (0, 1) entries we run gradient descent updates of the form DISPLAYFORM0 Furthermore, assume the number of parameters obey DISPLAYFORM1 with ??(C) the neural net cluster condition number per Definition 2.1.

Then, with probability at least 1 ??? 2 K 100 over randomly initialized W 0 DISPLAYFORM2 ??? N (0, 1), the iterates W ?? obey the following properties.??? The distance to initial point W 0 is upper bounded by DISPLAYFORM3 ??? After ?? ??? ?? 0 ???= c DISPLAYFORM4 the entrywise predictions of the learned network with respect to the ground truth labels DISPLAYFORM5 for all 1 ??? i ??? n. Furthermore, if the noise level ?? obeys ?? ??? ?? 8 the network predicts the correct label for all samples i.e.arg min DISPLAYFORM6 This result shows that in the limit 0 ??? 0 where the data points are perfectly clustered, the required amount of overparameterization can be reduced from kd ??? K 4 to kd ??? K 2 .

In this sense this can be thought of a nontrivial analogue of (5) where the number of data points are replaced with the number of clusters and the condition number of the data points is replaced with a cluster condition number.

This can be interpreted as ensuring that the network has enough capacity to fit the cluster centers {c } K =1 and the associated true labels.

Interestingly, the robustness benefits continue to hold in this case.

However, in this perfectly clustered scenario there is no need for early stopping and a robust network is trained as soon as the number of iterations are sufficiently large.

Infact, in this case given the clustered nature of the input data the network never overfits to the corrupted data even after many iterations.

B. To (over)fit to corrupted labels requires straying far from initializationIn this section we wish to provide further insight into why early stopping enables robustness and generalizable solutions.

Our main insight is that while a neural network maybe expressive enough to fit a corrupted dataset, the model has to travel a longer distance from the point of initialization as a function of the distance from the cluster centers ?? 0 and the amount of corruption.

We formalize this idea as follows.

Suppose 1.

two input points are close to each other (e.g. they are from the same cluster), 2.

but their labels are different, hence the network has to map them to distant outputs.

Then, the network has to be large enough so that it can amplify the small input difference to create a large output difference.

Our first result formalizes this for a randomly initialized network.

Our random initialization picks W with i.i.d.

standard normal entries which ensures that the network is isometric i.e. given input DISPLAYFORM7 ).Theorem B.1 Let x 1 , x 2 ??? R d be two vectors with unit Euclidean norm obeying DISPLAYFORM8 where v is fixed, W ??? R k??d , and k ??? cd with c > 0 a fixed constant.

Assume ?? ??? , ?? ?????? ??? ??. Let y 1 and y 2 be two scalars satisfying DISPLAYFORM9 ??? N (0, 1).

Then, with probability at least 1???2e DISPLAYFORM10 holds, we have DISPLAYFORM11 In words, this result shows that in order to fit to a data set with a single corrupted label, a randomly initialized network has to traverse a distance of at least ?? ?? 0 .

The next lemma clarifies the role of the corruption amount s and shows that more label corruption within a fixed class requires a model with a larger norm in order to fit the labels.

For this result we consider a randomized model with ?? 2 0 input noise variance.

DISPLAYFORM12 with labels y i = y and {x i } s i=1 with labels??? i =??? and assume these two labels are ?? separated i.e. y ?????? ??? ??.

Also suppose s ??? d and ?? DISPLAYFORM13 with probability at least 1 ??? e ???d 2 .Unlike Theorem E.15 this result lower bounds the network norm in lieu of the distance to the initialization W 0 .

However, using the triangular inequality we can in turn get a guarantee on the distance from initialization W 0 via triangle inequality as long as DISPLAYFORM14 The above Theorem implies that the model has to traverse a distance of at least DISPLAYFORM15 to perfectly fit corrupted labels.

In contrast, we note that the conclusions of the upper bound in Theorem 2.2 show that to be able to fit to the uncorrupted true labels the distance to initialization grows at most by ?? ?? 0 after ?? iterates.

This demonstrates that there is a gap in the required distance to initialization for fitting enough to generalize and overfitting.

To sum up, our results highlight that, one can find a network with good generalization capabilities and robustness to label corruption within a small neighborhood of the initialization and that the size of this neighborhood is independent of the corruption.

However, to fit to the corrupted labels, one has to travel much more, increasing the search space and likely decreasing generalization ability.

Thus, early stopping can enable robustness without overfitting by restricting the distance to the initialization.

In this section, we outline our approach to proving robustness of overparameterized neural networks.

Towards this goal, we consider a general formulation where we aim to fit a general nonlinear model of the form x ??? f (??, x) with ?? ??? R p denoting the parameters of the model.

For instance in the case of neural networks ?? represents its weights.

Given a data set of n input/label pairs {( DISPLAYFORM0 fit to this data by minimizing a nonlinear least-squares loss of the form DISPLAYFORM1 which can also be written in the more compact form DISPLAYFORM2 To solve this problem we run gradient descent iterations with a constant learning rate ?? starting from an initial point ?? 0 .

These iterations take the form DISPLAYFORM3 Here, J (??) is the n ?? p Jacobian matrix associated with the nonlinear mapping f defined via DISPLAYFORM4 Our approach is based on the hypothesis that the nonlinear model has a Jacobian matrix with bimodal spectrum where few singular values are large and remaining singular values are small.

This assumption is inspired by the fact that realistic datasets are clusterable in a proper, possibly nonlinear, representation space.

Indeed, one may argue that one reason for using neural networks is to automate the learning of such a representation (essentially the input to the softmax layer).

We formalize the notion of bimodal spectrum below.

Assumption 1 (Bimodal Jacobian) Let ?? ??? ?? ??? > 0 be scalars.

Let f ??? R p ??? R n be a nonlinear mapping and consider a set D ??? R p containing the initial point ?? 0 (i.e. ?? 0 ??? D).

Let S + ??? R n be a subspace and S ??? be its complement.

We say the mapping f has a Bimodal Jacobian with respect to the complementary subpspaces S + and S ??? as long as the following two assumptions hold for all ?? ??? D.??? Spectrum over S + : For all v ??? S + with unit Euclidian norm we have DISPLAYFORM5 ??? Spectrum over S ??? : For all v ??? S ??? with unit Euclidian norm we have J DISPLAYFORM6 We will refer to S + as the signal subspace and S ??? as the noise subspace.

When << ?? the Jacobian is approximately low-rank.

An extreme special case of this assumption is where = 0 so that the Jacobian matrix is exactly low-rank.

We formalize this assumption below for later reference .

387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 432 433 434 435 436 437 438 439 Assumption 2 (Low-rank Jacobian) Let ?? ??? ?? > 0 be scalars.

Consider a set D ??? R p containing the initial point ?? 0 (i.e. ?? 0 ??? D).

Let S + ??? R n be a subspace and S ??? be its complement.

For all ?? ??? D, v ??? S + and w ??? S ??? with unit Euclidian norm, we have that DISPLAYFORM7 Our dataset model in Definition 1.2 naturally has a lowrank Jacobian when 0 = 0 and each input example is equal to one of the K cluster centers {c } K =1 .

In this case, the Jacobian will be at most rank K since each row will be in the span of DISPLAYFORM8 .

The subspace S + is dictated by the membership of each cluster as follows: Let ?? ??? {1, . . .

, n} be the set of coordinates i such that x i = c .

Then, subspace is characterized by DISPLAYFORM9 When 0 > 0 and the data points of each cluster are not the same as the cluster center we have the bimodal Jacobian structure of Assumption 1 where over S ??? the spectral norm is small but nonzero.

In Section D, we verify that the Jacobian matrix of real datasets indeed have a bimodal structure i.e. there are few large singular values and the remaining singular values are small which further motivate Assumption 2.

This is inline with earlier papers which observed that Hessian matrices of deep networks have bimodal spectrum (approximately lowrank) (11) and is related to various results demonstrating that there are flat directions in the loss landscape (12).

Define the n-dimensional residual vector r where DISPLAYFORM0 A key idea in our approach is that we argue that (1) in the absence of any corruption r(??) approximately lies on the subspace S + and (2) if the labels are corrupted by a vector e, then e approximately lies on the complement space.

Before we state our general result we need to discuss another assumption and definition.

The Jacobian mapping J (??) associated to a nonlinear mapping DISPLAYFORM0 Additionally, to connect our results to the number of corrupted labels, we introduce the notion of subspace diffusedness defined below.

4 Note that, if DISPLAYFORM1 is continuous, the smoothness condition holds over any compact domain (albeit for a possibly large L).

DISPLAYFORM2 The following theorem is our meta result on the robustness of gradient descent to sparse corruptions on the labels when the Jacobian mapping is exactly low-rank.

Theorem E.14 for the perfectly clustered data ( 0 = 0) is obtained by combining this result with specific estimates developed for neural networks.

around an initial point ?? 0 and y = [y 1 . . .

y n ] ??? R n denoting the corrupted labels.

Also let??? = [??? 1 . .

.??? n ] ??? R n denote the uncorrupted labels and e = y ?????? the corruption.

Furthermore, suppose the initial residual f (?? 0 ) ?????? with respect to the uncorrupted labels obey f (?? 0 ) ?????? ??? S + .

Then, running gradient descent updates of the from (C.1) with a learning rate DISPLAYFORM3 Furthermore, assume ?? > 0 is a precision level obeying ?? ??? ?? S+ (e) ??? .

Then, after ?? ??? 5 ???? 2 log r0 2 ?? iterations, ?? ?? achieves the following error bound with respect to the true labels DISPLAYFORM4 Furthermore, if e has at most s nonzeros and S + is ?? diffused per Definition C.1, then using DISPLAYFORM5 This result shows that when the Jacobian of the nonlinear mapping is low-rank, gradient descent enjoys two intriguing properties.

First, gradient descent iterations remain rather close to the initial point.

Second, the estimated labels of the algorithm enjoy sample-wise robustness guarantees in the sense that the noise in the estimated labels are gracefully distributed over the dataset and the effects on individual label estimates are negligible.

This theorem is the key result that allows us to prove Theorem E.14 when the data points are perfectly clustered ( 0 = 0).

Furthermore, this theorem when combined with a perturbation analysis allows us to deal with data that is not perfectly clustered ( 0 > 0) and to conclude that with early stopping neural networks are rather robust to label corruption (Theorem 2.2).

441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 486 487 488 489 490 491 492 493 494 Finally, we note that a few recent publication (7; 9; 13) require the Jacobian to be well-conditioned to fit labels perfectly.

In contrast, our low-rank model cannot perfectly fit the corrupted labels.

Furthermore, when the Jacobian is bimodal (as seems to be the case for many practical data sets and neural network models) it would take a very long time to perfectly fit the labels and as demonstrated earlier such a model does not generalize and is not robust to corruptions.

Instead we focus on proving robustness with early stopping.

C.3.

To (over)fit to corrupted labels requires straying far from initializationIn this section we state a result that provides further justification as to why early stopping of gradient descent leads to more robust models without overfitting to corrupted labels.

This is based on the observation that while finding an estimate that fits the uncorrupted labels one does not have to move far from the initial estimate in the presence of corruption one has to stray rather far from the initialization with the distance from initialization increasing further in the presence of more corruption.

We make this observation rigorous below by showing that it is more difficult to fit to the portion of the residual that lies on the noise space compared to the portion on the signal space (assuming ?? ??? ).Theorem C.3 Denote the residual at initialization ?? 0 by r 0 = f (?? 0 ) ???

y. Define the residual projection over the signal and noise space as DISPLAYFORM6 Suppose Assumption 1 holds over an Euclidian ball D of radius R < max DISPLAYFORM7 around the initial point ?? 0 with ?? ??? .

Then, over D there exists no ?? that achieves zero training loss.

In particular, if D = R p , any parameter ?? achieving zero training loss (f (??) = y) satisfies the distance bound DISPLAYFORM8 This theorem shows that the higher the corruption (and hence E ??? ) the further the iterates need to stray from the initial model to fit the corrupted data.

We conduct several experiments to investigate the robustness capabilities of deep networks to label corruption.

In our first set of experiments, we explore the relationship between loss, accuracy, and amount of label corruption on the MNIST dataset to corroborate our theory.

Our next experiments study the distribution of the loss and the Jacobian on the CIFAR-10 dataset.

Finally, we simulate our theoretical model by generating data according to the corrupted data In FIG6 , we train the same model used in FIG1 with n = 3, 000 MNIST samples for different amounts of corruption.

Our theory predicts that more label corruption leads to a larger distance to initialization.

To probe this hypothesis, FIG6 and 3b visualizes training accuracy and training loss as a function of the distance from the initialization.

These results demonstrate that the distance from initialization gracefully increase with more corruption.

Next, we study the distribution of the individual sample losses on the CIFAR-10 dataset.

We conducted two experiments using Resnet-20 with cross entropy loss 5 .

In FIG8 we assess the noise robustness of gradient descent where we used all 50,000 samples with either 30% random corruption 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 540 541 542 543 544 545 546 547 548 or 50% random corruption.

Theorem E.14 predicts that when the corruption level is small, the loss distribution of corrupted vs clean samples should be separable.

FIG8 shows that when 30% of the data is corrupted the distributions are approximately separable.

When we increase the shuffling amount to 50% the training loss on the clean data increases as predicted by our theory and the distributions start to gracefully overlap.

As described in Section C, our technical framework utilizes a bimodal prior on the Jacobian matrix (C.2) of the model.

We now further investigate this hypothesis.

For a multiclass task, the Jacobian matrix is essentially a 3-way tensor where dimensions are sample size (n), total number of parameters in the model (p), and the number of classes (K).

The neural network model we used for CIFAR 10 has around 270,000 parameters in total.

In FIG9 we illustrate the singular value spectrum of the two multiclass Jacobian models where # >0.1?? top singular At initialization After training All classes 4 14 Correct class 15 16 Table 1 .

Jacobian of the network has few singular values that are significantly large i.e. larger than 0.1?? the spectral norm.

This is true whether we consider the initial network or final network.we form the Jacobian from all layers except the five largest (in total we usep ??? 90, 000 parameters).

6 We train the model with all samples and focus on the spectrum before and after the training.

In FIG9 , we picked n = 1000 samples and unfolded this tensor along parameters to obtain a 10, 000 ?? 90, 000 matrix which verifies our intuition on bimodality.

In particular, only 10 to 20 singular values are larger than 0.1?? the top one.

This is consistent with earlier works that studied the Hessian spectrum.

However, focusing on the Jacobian has the added advantage of requiring only first order information (11; 14).

A disadvantage is that the size of Jacobian grows with number of classes.

Intuitively, cross entropy loss focuses on the class associated with the label hence in FIG9 , we only picked the partial derivative associated with the correct class so that each sample is responsible for a single (sizep) vector.

This allowed us to scale to n = 10000 samples and the corresponding spectrum is strikingly similar.

Another intriguing finding is that the spectrums of before and after training are fairly close to each other highlighting that even at random initialization, spectrum is bimodal.

In FIG10 , we turn our attention to verifying our findings for the corrupted dataset model of Definition 1.2.

We generated K = 2 classes where the associated clusters centers are generated uniformly at random on the unit sphere of R d=20 .

We also generate the input samples at random around these two clusters uniformly at random on a sphere of radius ?? 0 = 0.5 around the corresponding cluster center.

Hence, the clusters are guaranteed to be at least 1 distance from each other to prevent overlap.

Overall we generate n = 400 samples (200 per class/cluster).

Here,K = K = 2 and the class labels are 0 and 1.

We picked a network with k = 1000 hidden units and trained on a data set with 400 samples where 30% of the labels were corrupted.

FIG10 plots the trajectory of training error and highlights the model achieves good classification in the first few iterations and ends up overfitting later on.

In FIG10 , we focus on the loss distribution of 6a at iterations 80 and 4500.

In this figure, we visualize the loss distribution of clean and corrupted data.

FIG10 highlights the loss distribution with early stopping and implies that the gap between corrupted and clean loss distributions is surprisingly resilient despite a large amount of corruption and the high-550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 594 595 596 597 598 599 600 601 602 603 capacity of the model.

In FIG10 , we repeat plot after many more iterations at which point the model overfits.

This plot shows that the distribution of the two classes overlap demonstrating that the model has overfit the corruption and lacks generalization/robustness.

We begin by defining the average Jacobian which will be used throughout our analysis.

Definition E.1 (Average Jacobian) We define the average Jacobian along the path connecting two points x, y ??? R p as J (y, x) ???= .

We experiment with the corrupted dataset model of Definition 1.2.

We picked K = 2 classes and set n = 400 and ??0 = 0.5.

Trained 30% corrupted data with k = 1000 hidden units.

Each corruption has 50% chance to remain in the correct class hence around 15% of the labels are actually flipped which corresponds to the dashed green line.

DISPLAYFORM0 The residualsr = f (??) ??? y, r = f (??) ??? y obey the following equationr = (I ??? ??C(??))r.

Proof Following Definition E.1, denoting f (??) ??? y =r and f (??) ??? y = r, we find that DISPLAYFORM1 Here (a) uses the fact that Jacobian is the derivative of f and (b) uses the fact that ???L(??) = J (??) T r.

Using Assumption C.1, one can show that sparse vectors have small projection on S + .Lemma E.3 Suppose Assumption C.1 holds.

If r ??? R n is a vector with s nonzero entries, we have that DISPLAYFORM2 Proof First, we bound the 2 projection of r on S + as follows DISPLAYFORM3 where we used the fact that v i ??? ??? ?? v 2 ??? n. Next, we conclude with DISPLAYFORM4 Proof The proof will be done inductively over the properties of gradient descent iterates and is inspired from the recent work (13).

In particular, (13) requires a well-conditioned Jacobian to fit labels perfectly.

In contrast, we have a lowrank Jacobian model which cannot fit the noisy labels (or it would have trouble fitting if the Jacobian was approximately low-rank).

Despite this, we wish to prove that gradient descent satisfies desirable properties such as robustness and closeness to initialization.

Let us introduce the notation related to the residual.

Set r ?? = f (?? ?? ) ??? y and let r 0 = f (?? 0 )???y be the initial residual.

We keep track of the growth of the residual by partitioning the residual as r ?? =r ?? +?? ?? where?? ?? = ?? S??? (r ?? ) ,r ?? = ?? S+ (r ?? ).

We claim that for all iterations ?? ??? 0, the following conditions hold.?? DISPLAYFORM5 Assuming these conditions hold till some ?? > 0, inductively, we focus on iteration ?? + 1.

First, note that these conditions imply that for all ?? ??? i ??? 0, ?? i ??? D where D is the Euclidian ball around ?? 0 of radius DISPLAYFORM6 .

This directly follows from (E.6) induction hypothesis.

Next, we claim that ?? ?? +1 is still within the set D. This can be seen as follows: DISPLAYFORM7 Proof Since range space of Jacobian is in S + and ?? ??? 1 ?? 2 , we begin by noting that DISPLAYFORM8 In the above, (a) follows from the fact that row range space of Jacobian is subset of S + via Assumption 2.

(b) follows from the definition ofr ?? .

(c) follows from the upper bound on the spectral norm of the Jacobian over D per Assumption 2, (d) from the fact that ?? ??? 1 ?? 2 , (e) from ?? ??? ??.

The latter combined with the triangular inequality and induction hypothesis (E.6) yields (after scaling (E.6) by 4 ??) DISPLAYFORM9 concluding the proof of ?? ?? +1 ??? D.To proceed, we shall verify that (E.6) holds for ?? + 1 as well.

Note that, following Lemma E.2, gradient descent iterate can be written as DISPLAYFORM10 Since both column and row space of C(?? ?? ) is subset of S + , we have that?? DISPLAYFORM11 This shows the first statement of the induction.

Next, over S + , we hav?? DISPLAYFORM12 where the second line uses the fact that?? ?? ??? S ??? and last line uses the fact thatr ?? ??? S + .

To proceed, we need to prove that C(?? ?? ) has desirable properties over S + , in particular, it contracts this space.

662 663 664 665 666 667 668 669 670 671 672 673 674 675 676 677 678 679 680 681 682 683 684 685 686 687 688 689 690 691 692 693 694 695 696 697 698 699 700 701 702 703 704 705 706 707 708 709 710 711 712 713 714 Claim 2 let P S+ ??? R n??n be the projection matrix to S + i.e. it is a positive semi-definite matrix whose eigenvectors over S + is 1 and its complement is 0.

Under the induction hypothesis and setup of the theorem, we have that DISPLAYFORM13 Proof The proof utilizes the upper bound on the learning rate.

The argument is similar to the proof of Lemma 9.7 of (13).

Suppose Assumption 3 holds.

Then, for any ?? 1 , ?? 2 ??? D we have DISPLAYFORM14 where for (a) we utilized the induction hypothesis (E.6) and (b) follows from the upper bound on ??.

Now that (E.22) is established, using following lemma, we find DISPLAYFORM15 The ?? 2 upper bound directly follows from Assumption 2 by again noticing range space of Jacobian is subset of S + .Lemma E.4 (Asymmetric PSD perturbation) Consider the matrices A, C ??? R n??p obeying A ??? C ??? ?? 2.

Also suppose CC T ??? ?? 2 P S+ .

Furthermore, assume range spaces of A, C lies in S + .

Then, DISPLAYFORM16 7 We say A ??? B if A ??? B is a positive semi-definite matrix in the sense that for any real vector v, v DISPLAYFORM17 Proof For r ??? S + with unit Euclidian norm, we have DISPLAYFORM18 Also, for any r, by range space assumption r T AC T r = ?? S+ (r)T AC T ?? S+ (r) (same for CC T ).

Combined with above, this concludes the claim.

What remains is proving the final two statements of the induction (E.6).

Note that, using the claim above and recalling (E.19) and using the fact that J (?? ?? +1 , ?? ?? ) ??? ??, the residual satisfies DISPLAYFORM19 where we used the fact that ?? ??? 1 2?? 2 .

Now, using the fact DISPLAYFORM20 which establishes the second statement of the induction (E.6).

What remains is obtaining the last statement of (E.6).To address this, completing squares, observe that DISPLAYFORM21 On the other hand, the distance to initial point satisfies DISPLAYFORM22 1 4 ??) and using induction hypothesis (E.6), we find that DISPLAYFORM23 This establishes the final line of the induction and concludes the proof of the upper bound on ?? ?? ??? ?? 0 2 .

To proceed, we shall bound the infinity norm of the residual.

Using DISPLAYFORM24 (E.27) What remains is controlling r ?? ??? .

For this term, we shall use the naive upper bound r ?? 2 .

Using the rate of convergence of the algorithm (E.6), we have that DISPLAYFORM25 We wish the right hand side to be at most ?? > 0 where ?? ??? ?? S+ (e) ??? .

This implies that we need DISPLAYFORM26 To conclude, note that since DISPLAYFORM27 ), we find that r ?? ??? ??? r ?? 2 ??? ??, which guarantees DISPLAYFORM28 which is the advertised result.

If e is s sparse and S + is diffused, applying Lemma C.1 we have DISPLAYFORM29 Since Jacobian is derivative of f , we have that DISPLAYFORM30 Now, define the matrices J + = ?? S+ (J ) and J ??? = ?? S??? (J ).

Using Assumption 1, we bound the spectral norms via DISPLAYFORM31 To proceed, projecting the residual on S + , we find for any ?? with f (??) = y DISPLAYFORM32 The identical argument for S ??? yields ?? ??? ?? 0 2 ??? E??? .

Together this implies DISPLAYFORM33 If R is strictly smaller than right hand side, we reach a contradiction as ?? ??? D. If D = R p , we still find (E.30).This shows that if is small and E ??? is nonzero, gradient descent has to traverse a long distance to find a good model.

Intuitively, if the projection over the noise space indeed contains the label noise, we actually don't want to fit that.

Algorithmically, our idea fits the residual over the signal space and not worries about fitting over the noise space.

Approximately speaking, this intuition corresponds to the 2 regularized problem min DISPLAYFORM34 If we set R = E+ ?? , we can hope that solution will learn only the signal and does not overfit to the noise.

The next section builds on this intuition and formalizes our algorithmic guarantees.

Throughout, ?? min (???) denotes the smallest singular value of a given matrix.

We first introduce helpful definitions that will be used in our proofs.

DISPLAYFORM0 be an input dataset generated according to Definition 1.1.

Also let {x i } n i=1 be the associated cluster centers, that is,x i = c iff x i is from the th cluster.

We define the support subspace S + as a subspace of dimension K, dictated by the cluster membership as follows.

Let ?? ??? {1, . . .

, n} be the set of coordinates i such thatx i = c .

Then, S + is characterized by DISPLAYFORM1 The Jacobian of the learning problem (1.3), at a matrix W is denoted by J (W , X) ??? R n??kd and is given by DISPLAYFORM2 Here * denotes the Khatri-Rao product.

The following theorem is borrowed from (5) and characterizes three key properties of the neural network Jacobian.

These are smoothness, spectral norm, and minimum singular value at initialization which correspond to Lemmas 6.6, 6.7, and 6.8 in that paper.

Theorem E.7 (Jacobian Properties at Cluster Center) DISPLAYFORM3 The Jacobian mapping with respect to the input-to-hidden weights obey the following properties.??? Smoothness is bounded by DISPLAYFORM4 ??? Top singular value is bounded by J (W , X) ??? ?? X .??? Let C > 0 be an absolute constant.

As long as DISPLAYFORM5 At random Gaussian initialization W 0 ??? N (0, 1) k??d , with probability at least 1 ??? 1 K 100 , we have DISPLAYFORM6 In our case, the Jacobian is not well-conditioned.

However, it is pretty well-structured as described previously.

To proceed, given a matrix X ??? R n??d and a subspace S ??? R n , we define the minimum singular value of the matrix over this subspace by ?? min (X, S) which is defined as DISPLAYFORM7 Here, P S ??? R n??n is the projection operator to the subspace.

Hence, this definition essentially projects the matrix on S and then takes the minimum singular value over that projected subspace.

The following theorem states the properties of the Jacobian at a clusterable dataset.

Theorem E.8 (Jacobian Properties at Clusterable Dataset) Let input samples (x i ) n i=1 be generated according to (?? 0 , ??) clusterable dataset model of Definition 1.1 and define DISPLAYFORM8 T .

Let S + be the support space and DISPLAYFORM9 be the associated clean dataset as described by Definition E.5.

DISPLAYFORM10 The Jacobian mapping at X with respect to the input-to-hidden weights obey the following properties.??? Smoothness is bounded by DISPLAYFORM11 ??? Top singular value is bounded by DISPLAYFORM12 ??? As long as DISPLAYFORM13 At random Gaussian initialization W 0 ??? N (0, 1) k??d , with probability at least 1 ??? 1 K 100 , we have DISPLAYFORM14 ??? The range space obeys range(J (W 0 ,X)) ??? S + where S + is given by Definition E.5.Proof Let J (W , C) be the Jacobian at the cluster center matrix.

Applying Theorem E.7, this matrix already obeys the properties described in the conclusions of this theorem with desired probability (for the last conclusion).

We prove our theorem by relating the cluster center Jacobian to the clean dataset Jacobian matrix J (W ,X).Note thatX is obtained by duplicating the rows of the cluster center matrix C. This implies that J (W ,X) is obtained by duplicating the rows of the cluster center Jacobian.

The critical observation is that, by construction in Definition 1.1, each row is duplicated somewhere between c low n K and c up n K.To proceed, fix a vector v and letp = J (W ,X)v ??? R n and p = J (W , C)v ??? R K .

Recall the definition of the support sets ?? from Definition E.5.

We have the identit??? DISPLAYFORM15 This impliesp ??? S + hence range(J (W ,X)) ??? S + .

Furthermore, the entries ofp repeats the entries of p somewhere between c low n K and c up n K. This implies that, DISPLAYFORM16 and establishes the upper and lower bounds on the singular values of J (W ,X) over S + in terms of the singular values of J (W , C).

Finally, the smoothness can be established similarly.

Given matrices W ,W , the rows of the difference DISPLAYFORM17 Hence the spectral norm is scaled by at most c up n K. k so that v 2 = 1.

Also assume we have n data points x 1 , x 2 , . . .

, x n ??? R d with unit euclidean norm ( x i 2 = 1) aggregated as rows of a matrix X ??? R n??d and the corresponding labels given by y ??? R n generated accoring to (??, ?? 0 = 0, ??) noisy dataset (Definition 1.2).

Then for W 0 ??? R k??d with i.i.d.

N (0, 1) entries DISPLAYFORM18 holds with probability at least 1 ??? K ???100 .Proof This lemma is based on a fairly straightforward union bound.

First, by construction y 2 ??? ??? n. What remains is bounding v T ?? W 0 X T 2 .

Since ?? 0 = 0 there are K unique rows.

We will show that each of the unique rows is bounded with probability 1 ??? K ???101 and union bounding will give the final result.

Let w be a row of W 0 and x be a row of X. Since ?? is ?? Lipschitz and DISPLAYFORM19 for some constant c > 0, concluding the proof.

E.2.1.

PROOF OF THEOREM E.14 We first prove a lemma regarding the projection of label noise on the cluster induced subspace.

DISPLAYFORM20 be an (??, ?? 0 = 0, ??) clusterable noisy dataset as described in Definition 1.2.

Let {??? i } n i=1be the corresponding noiseless labels.

Let J (W , C) be the Jacobian at the cluster center matrix which is rank K and S + be its column space.

Then, the difference between noiseless and noisy labels satisfy the bound DISPLAYFORM21 Proof Let e = y??????.

Observe that by assumption, th cluster has at most s = ??n errors.

Let I denote the membership associated with cluster i.e. I ??? {1, . . .

, n} and i ??? I if and only if x i belongs to th cluster.

Let 1( ) ??? R n be the indicator function of the th class where ith entry is 1 if i ??? I and 0 else for 1 ???

i ??? n.

Then, denoting the size of the th cluster by n , the projection to subspace S + can be written as the P matrix where DISPLAYFORM22 Let e be the error pattern associated with th cluster i.e. e is equal to e over I and zero outside.

Since cluster membership is non-overlapping, we have that DISPLAYFORM23 Similarly since supports of 1( ) are non-overlapping, we have that DISPLAYFORM24 Now, using e ??? ??? 2 (max distance between two labels), observe that DISPLAYFORM25 Since number of errors within cluster is at most n ??, we find that DISPLAYFORM26 C where L is the Lipschitz constant of Jacobian spectrum.

Denote r ?? = f (W ?? ) ??? y. Using Lemma E.9 with probability 1 ??? K ???100 , we have that r 0 2 = y ??? f (W 0 ) 2 ??? ?? c 0 n log K 128 for some c 0 > 0.

Corollary E.8 guarantees a uniform bound for ??, hence in Assumption 2, we pick DISPLAYFORM27 We shall also pick the minimum singular value over S + to be DISPLAYFORM28 We wish to verify Assumption 2 over the radius of DISPLAYFORM29 neighborhood of W 0 .

What remains is ensuring that Jacobian over S + is lower bounded by ??.

Our choice of k guarantees that at the initialization, with probability 1 ??? K ???100 , we have DISPLAYFORM30 Suppose LR ??? ?? = ?? 0 2.

Using triangle inequality on Jacobian spectrum, for any W ??? D, using W ??? W 0 F ??? R, we would have DISPLAYFORM31 Finally, since LR = 4L r 0 2 ?? ??? ??, the learning rate is DISPLAYFORM32 Overall, the assumptions of Theorem C.2 holds with stated ??, ??, L with probability 1 ??? 2K ???100 (union bounding initial residual and minimum singular value events).

This implies for all ?? > 0 the distance of current iterate to initial obeys DISPLAYFORM33 The final step is the properties of the label corruption.

Using Lemma E.10, we find that DISPLAYFORM34 Substituting the values corresponding to ??, ??, L yields that, for all gradient iterations with DISPLAYFORM35 denoting the clean labels by??? and applying Theorem C.2, we have that, the infinity norm of the residual obeys (using DISPLAYFORM36 This implies that if ?? ??? ?? 8, the network will miss the correct label by at most ?? 2, hence all labels (including noisy ones) will be correctly classified.

Consider DISPLAYFORM0 has mean zero.

Hence, using the fact that weighted sum of subGaussian random variables are subgaussian combined with (G.2) we conclude that DISPLAYFORM1 with probability at least 1 ??? e ??? t 2 2 .

Now combining (G.1) and (G.3) we have DISPLAYFORM2 with high probability.

Denote average neural net Jacobian at data X via DISPLAYFORM0 T be the input matrix obtained from Definition 1.1.

LetX be the noiseless inputs wherex i is the cluster center corresponding to x i .

Given weight matrices W 1 , W 2 ,W 1 ,W 2 , we have that DISPLAYFORM1 We first bound DISPLAYFORM2 To proceed, we use the results on the spectrum of Hadamard product of matrices due to Schur (15).

Given A ??? R k??d , B ??? R n??d matrices where B has unit length rows, we have DISPLAYFORM3 Secondly, DISPLAYFORM4 where reusing Schur's result and boundedness of ?? DISPLAYFORM5 Combining both estimates yields DISPLAYFORM6 To get the result on DISPLAYFORM7 according to (??, ?? 0 , ??) noisy dataset model and form the concatenated input/labels X ??? R d??n , y ??? R n .

LetX be the clean input sample matrix obtained by mapping x i to its associated cluster center.

Set learning rate ?? ??? K 2cupn?? 2 C 2 and maximum iterations ?? 0 satisfying DISPLAYFORM8 where C 1 ??? 1 is a constant of our choice.

Suppose input noise level ?? 0 and number of hidden nodes obey DISPLAYFORM9 ??? N (0, 1).

Starting from W 0 =W 0 consider the gradient descent iterations over the losses DISPLAYFORM10 Then, for all gradient descent iterations satisfying ?? ??? ?? 0 , we have that 1103 1104 1105 1106 1107 1108 1109 1110 1111 1112 1113 1114 1115 1116 1117 1118 1119 1120 1121 1122 1123 1124 1125 1126 1127 1128 1129 1130 1131 1132 1133 1134 1135 1136 1137 1138 1139 1140 1141 1142 1143 1144 1145 1146 1147 1148 1149 1150 1151 1152 1153 1154 The proof is by induction.

Suppose it holds until t ??? ?? 0 ??? 1.

At t + 1, via (E.37) we have that DISPLAYFORM11 DISPLAYFORM12 Right hand side holds since L ??? 1 2????0??.

This establishes the induction for d t+1 .Next, we show the induction on p t .

Observe that 3d t +d t+1 ??? 10?? 0 ???? ??? n?? 0 ??(1 + 8???? 0 ?? 2 ).

Following (E.39) and using DISPLAYFORM13 Concluding the induction since L satisfies the final line.

Consequently, for all 0 ??? t ??? ?? 0 , we have that DISPLAYFORM14 which is implied by k ??? O(?? 10 K 2 C 4 ??(C) 4 log( DISPLAYFORM15 ).Finally, following (E.40), distance satisfies DISPLAYFORM16 ).E.3.1.

COMPLETING THE PROOF OF THEOREM 2.2 Theorem 2.2 is obtained by the theorem below when we ignore the log terms, and treating ??, ??(C) as constants.

We also plug in ?? = K 2cupn?? 2 C 2 .Theorem E.13 (Training neural nets with corrupted labels) Let {(x i , y i )} n i=1 be an (s, ?? 0 , ??) clusterable noisy dataset as described in Definition 1.2.

Let {??? i } n i=1 be the corresponding noiseless labels.

Suppose ??(0) , ?? ??? , ?? ?????? ??? ?? for some ?? ??? 1, input noise and the number of hidden nodes satisfy ?? 0 ???O( ??(C) DISPLAYFORM17 )

.where C ??? R K??d is the matrix of cluster centers.

Set learning rate ?? ??? ??? N (0, 1).

With probability 1 ??? 3 K 100 , after DISPLAYFORM18 ) log( ?? n log K ?? ) iterations, for all 1 ??? i ??? n,we have that??? The per sample normalized 2 norm bound satisfies DISPLAYFORM19 ??? Suppose ?? ??? ?? 8.

Denote the total number of prediction errors with respect to true labels (i.e. not satisfying (E.46)) by err(W ).

With same probability, err(W ?? ) obeys DISPLAYFORM20 log( ?? ??? n log K ?? ).???

Suppose ?? ??? ?? 8 and ?? 0 ??? c DISPLAYFORM21 , then, W ?? assigns all input samples x i to correct ground truth labels??? i i.e. (E.46) holds for all 1 ??? i ??? n.??? Finally, for any iteration count 0 ??? t ??? ?? the total distance to initialization is bounded as DISPLAYFORM22 .

Proof Note that proposed number of iterations ?? is set so that it is large enough for Theorem E.14 to achieve small error in the clean input model (?? 0 = 0) and it is small enough so that Theorem E.12 is applicable.

In light of Theorems E.12 and E.14 consider two gradient descent iterations starting from W 0 where one uses clean dataset (as if input vectors are perfectly cluster centers)X and other uses the 1157 1158 1159 1160 1161 1162 1163 1164 1165 1166 1167 1168 1169 1170 1171 1172 1173 1174 1175 1176 1177 1178 1179 1180 1181 1182 1183 1184 1185 1186 1187 1188 1189 1190 1191 1192 1193 1194 1195 1196 1197 1198 1199 1200 1201 1202 1203 1204 1205 1206 1207 1208 1209 original dataset X. Denote the prediction residual vectors of the noiseless and original problems at time ?? with respect true ground truth labels??? byr ?? = f (W ?? ,X) ?????? and r ?? = f (W ?? , X) ?????? respectively.

Applying Theorems E.12 and E.14, under the stated conditions, we have that r ?? ??? ??? 4?? and (E.42) DISPLAYFORM0 First statement: The latter two results imply the 2 error bounds on r ?? = f (W ?? , X) ??????.

Second statement: To assess the classification rate we count the number of entries of r ?? = f (W ?? , X) ?????? that is larger than the class margin ?? 2 in absolute value.

Suppose ?? ??? ?? 8.

Let I be the set of entries obeying this.

For i ??? I using r ?? ??? ??? 4?? ??? ?? 4, we have r ??,i ??? ?? 2 ??? r ??,i + r ??,i ???r ??,i ??? ?? 2 ??? r ??,i ???r ??,i ??? ?? 4.Consequently, we find that r ?? ???r ?? 1 ??? I ?? 4.Converting 2 upper bound on the left hand side to 1 , we obtain c ??? n ?? 0 ?? 3 K ??? n log K ??(C) log( ?? ??? n log K ?? )

??? I ?? 4.Hence, the total number of errors is at most DISPLAYFORM1 Third statement -Showing zero error:

Pick an input sample x from dataset and its clean versionx.

We will argue that f (W ?? , x) ??? f (W ?? ,x) is smaller than ?? 4 when ?? 0 is small enough.

We again write DISPLAYFORM2 The first term can be bounded via DISPLAYFORM3 Next, we need to bound DISPLAYFORM4 where DISPLAYFORM5 ), x ???x 2 ??? ?? 0 and W 0 i.i.d.??? N (0, I).

Consequently, using by assumption we have DISPLAYFORM6 and applying an argument similar to Theorem E.15 (detailed in Appendix G), with probability at 1 ??? 1 n 100 , we find that DISPLAYFORM7 Combining the two bounds above we get DISPLAYFORM8 Hence, if ?? 0 ??? c DISPLAYFORM9 , we obtain that, for all DISPLAYFORM10 If ?? ??? ?? 8, we obtain f (W ?? , x i ) ?????? i < ?? 2 hence, W ?? outputs the correct decision for all samples.

Fourth statement -Distance: This follows from the triangle inequality DISPLAYFORM11 We have that right hand side terms are at most O(?? K log K ??(C)) and O(t???? 0 ?? 4 Kn ??(C) log( DISPLAYFORM12 Theorems E.12 and E.14 respectively.

This implies (E.41).Before we end this section we would like to note that in the limit of 0 ??? 0 where the input data set is perfectly clustered one can improve the amount of overparamterization.

Indeed, the result above is obtained via a perturbation argument from this more refined result stated below .

1212 1213 1214 1215 1216 1217 1218 1219 1220 1221 1222 1223 1224 1225 1226 1227 1228 1229 1230 1231 1232 1233 1234 1235 1236 1237 1238 1239 1240 1241 1242 1243 1244 1245 1246 1247 1248 1249 1250 1251 1252 1253 1254 1255 1256 1257 1258 1259 1260 1261 1262 1263

<|TLDR|>

@highlight

We prove that gradient descent is robust to label corruption despite over-parameterization under a rich dataset model.