RMSProp and ADAM continue to be extremely popular algorithms for training neural nets but their theoretical convergence properties have remained unclear.

Further, recent work has seemed to suggest that these algorithms have worse generalization properties when compared to carefully tuned stochastic gradient descent or its momentum variants.

In this work, we make progress towards a deeper understanding of ADAM and RMSProp in two ways.

First, we provide proofs that these adaptive gradient algorithms are guaranteed to reach criticality for smooth non-convex objectives, and we give bounds on the running time.



Next we design experiments to empirically study the convergence and generalization properties of RMSProp and ADAM against Nesterov's Accelerated Gradient method on a variety of common autoencoder setups and on VGG-9 with CIFAR-10.

Through these experiments we demonstrate the interesting sensitivity that ADAM has to its momentum parameter \beta_1.

We show that at very high values of the momentum parameter (\beta_1 = 0.99) ADAM outperforms a carefully tuned NAG on most of our experiments, in terms of getting lower training and test losses.

On the other hand, NAG can sometimes do better when ADAM's \beta_1 is set to the most commonly used value: \beta_1 = 0.9, indicating the importance of tuning the hyperparameters of ADAM to get better generalization performance.



We also report experiments on different autoencoders to demonstrate that NAG has better abilities in terms of reducing the gradient norms, and it also produces iterates which exhibit an increasing trend for the minimum eigenvalue of the Hessian of the loss function at the iterates.

Many optimization questions arising in machine learning can be cast as a finite sum optimization problem of the form: min x f (x) where f (x) = 1 k k i=1 f i (x).

Most neural network problems also fall under a similar structure where each function f i is typically non-convex.

A well-studied algorithm to solve such problems is Stochastic Gradient Descent (SGD), which uses updates of the form: x t+1 := x t ??? ?????f it (x t ), where ?? is a step size, andf it is a function chosen randomly from {f 1 , f 2 , . . .

, f k } at time t.

Often in neural networks, "momentum" is added to the SGD update to yield a two-step update process given as: v t+1 = ??v t ??? ?????f it (x t ) followed by x t+1 = x t + v t+1 .

This algorithm is typically called the Heavy-Ball (HB) method (or sometimes classical momentum), with ?? > 0 called the momentum parameter (Polyak, 1987) .

In the context of neural nets, another variant of SGD that is popular is Nesterov's Accelerated Gradient (NAG), which can also be thought of as a momentum method (Sutskever et al., 2013) , and has updates of the form v t+1 = ??v t ??? ?????f it (x t + ??v t ) followed by x t+1 = x t + v t+1 (see Algorithm 1 for more details).Momentum methods like HB and NAG have been shown to have superior convergence properties compared to gradient descent in the deterministic setting both for convex and non-convex functions (Nesterov, 1983; Polyak, 1987; Zavriev & Kostyuk, 1993; Ochs, 2016; O'Neill & Wright, 2017; Jin et al., 2017) .

While (to the best of our knowledge) there is no clear theoretical justification in the stochastic case of the benefits of NAG and HB over regular SGD in general (Yuan et al., 2016; Kidambi et al., 2018; Wiegerinck et al., 1994; Orr & Leen, 1994; Yang et al., 2016; Gadat et al., 2018) , unless considering specialized function classes (Loizou & Richt??rik, 2017) ; in practice, these momentum methods, and in particular NAG, have been repeatedly shown to have good convergence and generalization on a range of neural net problems (Sutskever et al., 2013; Lucas et al., 2018; Kidambi et al., 2018) .The performance of NAG (as well as HB and SGD), however, are typically quite sensitive to the selection of its hyper-parameters: step size, momentum and batch size (Sutskever et al., 2013) .

Thus, "adaptive gradient" algorithms such as RMSProp (Algorithm 2) (Tieleman & Hinton, 2012) and ADAM (Algorithm 3) (Kingma & Ba, 2014) have become very popular for optimizing deep neural networks (Melis et al., 2017; Xu et al., 2015; Denkowski & Neubig, 2017; Gregor et al., 2015; Radford et al., 2015; Bahar et al., 2017; Kiros et al., 2015) .

The reason for their widespread popularity seems to be the fact that they are believed to be easier to tune than SGD, NAG or HB.

Adaptive gradient methods use as their update direction a vector which is the image under a linear transformation (often called the "diagonal pre-conditioner") constructed out of the history of the gradients, of a linear combination of all the gradients seen till now.

It is generally believed that this "pre-conditioning" makes these algorithms much less sensitive to the selection of its hyperparameters.

A precursor to these RMSProp and ADAM was outlined in Duchi et al. (2011) .Despite their widespread use in neural net problems, adaptive gradients methods like RMSProp and ADAM lack theoretical justifications in the non-convex setting -even with exact/deterministic gradients (Bernstein et al., 2018) .

Further, there are also important motivations to study the behavior of these algorithms in the deterministic setting because of usecases where the amount of noise is controlled during optimization, either by using larger batches (Martens & Grosse, 2015; De et al., 2017; Babanezhad et al., 2015) or by employing variance-reducing techniques (Johnson & Zhang, 2013; Defazio et al., 2014) .Further, works like Wilson et al. (2017) and Keskar & Socher (2017) have shown cases where SGD (no momentum) and HB (classical momentum) generalize much better than RMSProp and ADAM with stochastic gradients.

Wilson et al. (2017) also show that ADAM generalizes poorly for large enough nets and that RMSProp generalizes better than ADAM on a couple of neural network tasks (most notably in the character-level language modeling task).

But in general it's not clear and no heuristics are known to the best of our knowledge to decide whether these insights about relative performances (generalization or training) between algorithms hold for other models or carry over to the full-batch setting.

A summary of our contributions In this work we try to shed some light on the above described open questions about adaptive gradient methods in the following two ways.??? To the best of our knowledge, this work gives the first convergence guarantees for adaptive gradient based standard neural-net training heuristics.

Specifically we show run-time bounds for deterministic RMSProp and ADAM to reach approximate criticality on smooth non-convex functions, as well as for stochastic RMSProp under an additional assumption.

Recently, Reddi et al. (2018) have shown in the setting of online convex optimization that there are certain sequences of convex functions where ADAM and RMSprop fail to converge to asymptotically zero average regret.

We contrast our findings with Theorem 3 in Reddi et al. (2018) .

Their counterexample for ADAM is constructed in the stochastic optimization framework and is incomparable to our result about deterministic ADAM.

Our proof of convergence to approximate critical points establishes a key conceptual point that for adaptive gradient algorithms one cannot transfer intuitions about convergence from online setups to their more common use case in offline setups.??? Our second contribution is empirical investigation into adaptive gradient methods, where our goals are different from what our theoretical results are probing.

We test the convergence and generalization properties of RMSProp and ADAM and we compare their performance against NAG on a variety of autoencoder experiments on MNIST data, in both full and mini-batch settings.

In the full-batch setting, we demonstrate that ADAM with very high values of the momentum parameter (?? 1 = 0.99) matches or outperforms carefully tuned NAG and RMSProp, in terms of getting lower training and test losses.

We show that as the autoencoder size keeps increasing, RMSProp fails to generalize pretty soon.

In the mini-batch experiments we see exactly the same behaviour for large enough nets.

We further validate this behavior on an image classification task on CIFAR-10 using a VGG-9 convolutional neural network, the results to which we present in the Appendix E. We note that recently it has been shown by Lucas et al. (2018) , that there are problems where NAG generalizes better than ADAM even after tuning ?? 1 .

In contrast our experiments reveal controlled setups where tuning ADAM's ?? 1 closer to 1 than usual practice helps close the generalization gap with NAG and HB which exists at standard values of ?? 1 .Remark.

Much after this work was completed we came to know of a related paper (Li & Orabona, 2018 ) which analyzes convergence rates of a modification of AdaGrad (not RMSPRop or ADAM).After the initial version of our work was made public, a few other analysis of adaptive gradient methods have also appeared like Chen et al. (2018) , Zhou et al. (2018) and Zaheer et al. (2018) .

Firstly we define the smoothness property that we assume in our proofs for all our non-convex objectives.

This is a standard assumption used in the optimization literature.

Definition 1.

L???smoothness If f : R d ??? R is at least once differentiable then we call it L???smooth for some L > 0 if for all x, y ??? R d the following inequality holds, DISPLAYFORM0 We need one more definition that of square-root of diagonal matrices, Definition 2.

Square root of the Penrose inverse DISPLAYFORM1 , where {e i } {i=1,...,d} is the standard basis of R d Now we list out the pseudocodes used for NAG, RMSProp and ADAM in theory and experiments, Nesterov's Accelerated Gradient (NAG) Algorithm Algorithm 1 NAG 1:

Input : A step size ??, momentum ?? ??? [0, 1), and an initial starting point x 1 ??? R d , and we are given query access to a (possibly noisy) oracle for gradients of f : DISPLAYFORM2 Initialize : v 1 = 0 4: DISPLAYFORM3 6: DISPLAYFORM4 end for 8: end function DISPLAYFORM5 , and we are given query access to a (possibly noisy) oracle for gradients of f : DISPLAYFORM6 Initialize : v 0 = 0 4: DISPLAYFORM7 8: end for 10: end function DISPLAYFORM8 DISPLAYFORM9 and we are given oracle access to the gradients of f : DISPLAYFORM10 Initialize : m 0 = 0, v 0 = 0 4: DISPLAYFORM11 9: DISPLAYFORM12 end for 11: end function

Previously it has been shown in Rangamani et al. (2017) that mini-batch RMSProp can off-the-shelf do autoencoding on depth 2 autoencoders trained on MNIST data while similar results using nonadaptive gradient descent methods requires much tuning of the step-size schedule.

Here we give the first result about convergence to criticality for stochastic RMSProp albeit under a certain technical assumption about the training set (and hence on the first order oracle).

Towards that we need the following definition, Definition 3.

The sign function We define the function sign : DISPLAYFORM0 DISPLAYFORM1 c) ?? f < ??? is an upperbound on the norm of the gradients of f i and (d) f has a minimizer, i.e., there exists x * such that f (x * ) = min x???R d f (x).

Let the gradient oracle be s.t when invoked at some x t ??? R d it uniformly at random picks i t ??? {1, 2, .., k} and returns, ???f it (x t ) = g t .

Then corresponding to any , ?? > 0 and a starting point x 1 for Algorithm 2, we can define, DISPLAYFORM2 (1?????2)?? s.t.

we are guaranteed that the iterates of Algorithm 2 using a constant step-length of, ?? = DISPLAYFORM3 will find an ???critical point in at most T steps in the sense that, min t=1,2..

DISPLAYFORM4 Remark.

We note that the theorem above continues to hold even if the constraint (b) that we have about the signs of the gradients of the {f i } i=1,...,k only holds on the points in R d that the stochastic RMSProp visits and its not necessary for the constraint to be true everywhere in the domain.

Further we can say in otherwords that this constraint ensures all the options for the gradient that this stochastic oracle has at any point, to lie in the same orthant of R d though this orthant itself can change from one iterate of the next.

A related result was concurrently shown by Zaheer et al. (2018) .Next we see that such sign conditions are not necessary to guarantee convergence of the deterministic RMSProp which corresponds to the full-batch RMSProp experiments in Section 5.3.

Theorem 3.2.

Convergence of deterministic RMSProp -the version with standard speeds (Proof in subsection A.2) Let f : R d ??? R be L???smooth and let ?? < ??? be an upperbound on the norm of the gradient of f .

Assume also that f has a minimizer, i.e., there exists x * such that f (x * ) = min x???R d f (x).

Then the following holds for Algorithm 2 with a deterministic gradient oracle:For any , ?? > 0, using a constant step length of DISPLAYFORM5 , where x 1 is the first iterate of the algorithm.

One might wonder if the ?? parameter introduced in the algorithms above is necessary to get convergence guarantees for RMSProp.

Towards that in the following theorem we show convergence of another variant of deterministic RMSProp which does not use the ?? parameter and instead uses other assumptions on the objective function and step size modulation.

But these tweaks to eliminate the need of ?? come at the cost of the convergence rates getting weaker.

Theorem 3.3.

Convergence of deterministic RMSProp -the version with no ?? shift (Proof in subsection A.3) Let f : R d ??? R be L???smooth and let ?? < ??? be an upperbound on the norm of the gradient of f .

Assume also that f has a minimizer, i.e., there exists x * such that f (x * ) = min x???R d f (x), and the function f be bounded from above and below by constants B and DISPLAYFORM6 the Algorithm 2 with a deterministic gradient oracle and ?? = 0 is guaranteed to reach a t-th iterate s.t.

1 ??? t ??? T and ???f (x t ) ??? .In Section 5.3 we show results of our experiments with full-batch ADAM.

Towards that, we analyze deterministic ADAM albeit in the small ?? 1 regime.

We note that a small ?? 1 does not cut-off contributions to the update direction from gradients in the arbitrarily far past (which are typically significantly large), and neither does it affect the non-triviality of the pre-conditioner which does not depend on ?? 1 at all.

Theorem 3.4.

Deterministic ADAM converges to criticality (Proof in subsection A.4) Let f : R d ??? R be L???smooth and let ?? < ??? be an upperbound on the norm of the gradient of f .

Assume also that f has a minimizer, i.e., there exists x * such that f (x * ) = min x???R d f (x).

Then the following holds for Algorithm 3: DISPLAYFORM7 , there exist step sizes ?? t > 0, t = 1, 2, . . .

and a natural number T (depending on ?? 1 , ??) such that ???f (x t ) ??? for some t ??? T .In particular if one sets ?? 1 = +2?? , ?? = 2??, and DISPLAYFORM8 3( +2??) 2 where g t is the gradient of the objective at the t th iterate, then T can be taken to be DISPLAYFORM9 , where x 2 is the second iterate of the algorithm.

Our motivations towards the above theorem were primarily rooted in trying to understand the situations where ADAM can converge at all (given the negative results about ADAM as in Reddi et al. (2018) ).

But we point out that it remains open to tighten the analysis of deterministic ADAM and obtain faster rates than what we have shown in the theorem above.

Remark.

It is sometimes believed that ADAM gains over RMSProp because of its "bias correction term" which refers to the step length of ADAM having an iteration dependence of the following form, 1 ??? ?? t 2 /(1 ??? ?? t 1 ).

In the above theorem, we note that the 1/(1 ??? ?? t 1 ) term of this "bias correction term" naturally comes out from theory!

4 EXPERIMENTAL SETUP For testing the empirical performance of ADAM and RMSProp, we perform experiments on fully connected autoencoders using ReLU activations and shared weights and on CIFAR-10 using VGG-9, a convolutional neural network.

Let z ??? R d be the input vector to the autoencoder, {W i } i=1,.., denote the weight matrices of the net and {b i } i=1,..,2 be the bias vectors.

Then the output??? ??? R d of the autoencoder is defined as??? DISPLAYFORM10 This defines an autoencoder with 2 ??? 1 hidden layers using weight matrices and 2 bias vectors.

Thus, the parameters of this model are given by DISPLAYFORM11 (where we imagine all vectors to be column vectors by default).

The loss function, for an input z is then given by: f (z; x) = z ?????? 2 .Such autoencoders are a fairly standard setup that have been used in previous work (Arpit et al., 2015; Baldi, 2012; Kuchaiev & Ginsburg, 2017; Vincent et al., 2010) .

There have been relatively fewer comparisons of ADAM and RMSProp with other methods on a regression setting.

We were motivated by Rangamani et al. (2017) who had undertaken a theoretical analysis of autoencoders and in their experiments had found RMSProp to have good reconstruction error for MNIST when used on even just 2 layer ReLU autoencoders.

To keep our experiments as controlled as possible, we make all layers in a network have the same width (which we denote as h).

Thus, given a size d for the input image, the weight matrices (as defined above) are given by: DISPLAYFORM12 . .

, .

This allowed us to study the effect of increasing depth or width h without having to deal with added confounding factors.

For all experiments, we use the standard "Glorot initialization" for the weights (Glorot & Bengio, 2010) , where each element in the weight matrix is initialized by sampling from a uniform distribution with [???limit, limit], limit = 6/(fan in + fan out ), where fan in denotes the number of input units in the weight matrix, and fan out denotes the number of output units in the weight matrix.

All bias vectors were initialized to zero.

No regularization was used.

We performed autoencoder experiments on the MNIST dataset for various network sizes (i.e., different values of and h).

We implemented all experiments using TensorFlow (Abadi et al., 2016) using an NVIDIA GeForce GTX 1080 Ti graphics card.

We compared the performance of ADAM and RMSProp with Nesterov's Accelerated Gradient (NAG).

All experiments were run for 10 5 iterations.

We tune over the hyper-parameters for each optimization algorithm using a grid search as described in the Appendix (Section B).

To pick the best set of hyper-parameters, we choose the ones corresponding to the lowest loss on the training set at the end of 10 5 iterations.

Further, to cut down on the computation time so that we can test a number of different neural net architectures, we crop the MNIST image from 28 ?? 28 down to a 22 ?? 22 image by removing 3 pixels from each side (almost all of which is whitespace).

We are interested in first comparing these algorithms in the full-batch setting.

To do this in a computationally feasible way, we consider a subset of the MNIST dataset (we call this: mini-MNIST), which we build by extracting the first 5500 images in the training set and first 1000 images in the test set in MNIST.

Thus, the training and testing datasets in mini-MNIST is 10% of the size of the MNIST dataset.

Thus the training set in mini-MNIST contains 5500 images, while the test set contains 1000 images.

This subset of the dataset is a fairly reasonable approximation of the full MNIST dataset (i.e., contains roughly the same distribution of labels as in the full MNIST dataset), and thus a legitimate dataset to optimize on.

To test if our conclusions on the full-batch case extend to the mini-batch case, we then perform the same experiments in a mini-batch setup where we fix the mini-batch size at 100.

For the mini-batch experiment, we consider the full training set of MNIST, instead of the mini-MNIST dataset considered for the full-batch experiments and we also test on CIFAR-10 using VGG-9, a convolutional neural network.

The ?? parameter is a feature of the default implementations of RMSProp and ADAM such as in TensorFlow.

Most interestingly this strictly positive parameter is crucial for our proofs.

In this section we present experimental evidence that attempts to clarify that this isn't merely a theoretical artefact but its value indeed has visible effect on the behaviours of these algorithms.

We see in FIG1 that on increasing the value of this fixed shift parameter ??, ADAM in particular, is strongly helped towards getting lower gradient norms and lower test losses though it can hurt its ability to get lower training losses.

The plots are shown for optimally tuned values for the other hyper-parameters.

To check whether NAG, ADAM or RMSProp is capable of consistently moving from a "bad" saddle point to a "good" saddle point region, we track the most negative eigenvalue of the Hessian ?? min (Hessian).

Even for a very small neural network with around 10 5 parameters, it is still intractable to store the full Hessian matrix in memory to compute the eigenvalues.

Instead, we use the Scipy library function scipy.sparse.linalg.eigsh that can use a function that computes the matrix-vector products to compute the eigenvalues of the matrix (Lehoucq et al., 1998) .

Thus, for finding the eigenvalues of the Hessian, it is sufficient to be able to do Hessian-vector products.

This can be done exactly in a fairly efficient way (Townsend, 2008) .We display a representative plot in FIG3 which shows that NAG in particular has a distinct ability to gradually, but consistently, keep increasing the minimum eigenvalue of the Hessian while continuing to decrease the gradient norm.

However unlike as in deeper autoencoders in this case the gradient norms are consistently bigger for NAG, compared to RMSProp and ADAM.

In contrast, RSMProp and ADAM quickly get to a high value of the minimum eigenvalue and a small gradient norm, but somewhat stagnate there.

In short, the trend looks better for NAG, but in actual numbers RMSProp and ADAM do better.

In Figure 3 , we show how the training loss, test loss and gradient norms vary through the iterations for RMSProp, ADAM (at ?? 1 = 0.9 and 0.99) and NAG (at ?? = 0.9 and 0.99) on a 3 hidden layer autoencoder with 1000 nodes in each hidden layer trained on mini-MNIST.

Appendix D.1 and D.2 have more such comparisons for various neural net architectures with varying depth and width and input image sizes, where the following qualitative results also extend.

Conclusions from the full-batch experiments of training autoencoders on mini-MNIST:??? Pushing ?? 1 closer to 1 significantly helps ADAM in getting lower training and test losses and at these values of ?? 1 , it has better performance on these metrics than all the other algorithms.

One sees cases like the one displayed in Figure 3 where ADAM at ?? 1 = 0.9 was getting comparable or slightly worse test and training errors than NAG.

But once ?? 1 gets closer to 1, ADAM's performance sharply improves and gets better than other algorithms.??? Increasing momentum helps NAG get lower gradient norms though on larger nets it might hurt its training or test performance.

NAG does seem to get the lowest gradient norms compared to the other algorithms, except for single hidden layer networks like in FIG3 .

In Figure 4 , we show how training loss, test loss and gradient norms vary when using mini-batches of size 100, on a 5 hidden layer autoencoder with 1000 nodes in each hidden layer trained on the full MNIST dataset.

The same phenomenon as here has been demonstrated in more such mini-batch comparisons on autoencoder architectures with varying depths and widths in Appendix D.3 and on VGG-9 with CIFAR-10 in Appendix E.Conclusions from the mini-batch experiments of training autoencoders on full MNIST dataset:??? Mini-batching does seem to help NAG do better than ADAM on small nets.

However, for larger nets, the full-batch behavior continues, i.e., when ADAM's momentum parameter ?? 1 is pushed closer to 1, it gets better generalization (significantly lower test losses) than NAG at any momentum tested.??? In general, for all metrics (test loss, training loss and gradient norm reduction) both ADAM as well as NAG seem to improve in performance when their momentum parameter (?? for NAG and ?? 1 for ADAM) is pushed closer to 1.

This effect, which was present in the full-batch setting, seems to get more pronounced here.??? As in the full-batch experiments, NAG continues to have the best ability to reduce gradient norms while for larger enough nets, ADAM at large momentum continues to have the best training error.

To the best of our knowledge, we present the first theoretical guarantees of convergence to criticality for the immensely popular algorithms RMSProp and ADAM in their most commonly used setting of optimizing a non-convex objective.

By our experiments, we have sought to shed light on the important topic of the interplay between adaptivity and momentum in training nets.

By choosing to study textbook autoencoder architectures where various parameters of the net can be changed controllably we highlight the following two aspects that (a) the value of the gradient shifting hyperparameter ?? has a significant influence on the performance of ADAM and RMSProp and (b) ADAM seems to perform particularly well (often supersedes Nesterov accelerated gradient method) when its momentum parameter ?? 1 is very close to 1.

On VGG-9 with CIFAR-10 and for the task of training autoencoders on MNIST we have verified these conclusions across different widths and depths of nets as well as in the full-batch and the mini-batch setting (with large nets) and under compression of the input/out image size.

Curiously enough, this regime of ?? 1 being close to 1 is currently not within the reach of our proof techniques of showing convergence for ADAM.

Our experiments give strong reasons to try to advance theory in this direction in future work.

Ilya Sutskever, James Martens, George Dahl, and Geoffrey Hinton.

Now we give the proof of Theorem 3.1.Proof.

We define ?? t := max k=1,..,t ???f i k (x k ) and we solve the recursion for v t as, DISPLAYFORM0 .

This lets us write the following bounds, DISPLAYFORM1 i and this lets us get the following bounds, DISPLAYFORM2 Now we invoke the bounded gradient assumption about the f i functions and replace in the above equation the eigenvalue bounds of the pre-conditioner by worst-case estimates ?? max and ?? min defined as, DISPLAYFORM3 Using the L-smoothness of f between consecutive iterates x t and x t+1 we have, DISPLAYFORM4 We note that the update step of stochastic RMSProp is x t+1 = x t ??? ??(V t ) ??? 1 2 g t where g t is the stochastic gradient at iterate x t .

Let H t = {x 1 , x 2 , .., x t } be the set of random variables corresponding to the first t iterates.

The assumptions we have about the stochastic oracle give us the following relations, DISPLAYFORM5 f .

Now we can invoke these stochastic oracle's properties and take a conditional (on H t ) expectation over g t of the L???smoothness in equation to get, DISPLAYFORM6 We now separately analyze the middle term in the RHS above in Lemma A.1 below and we get, DISPLAYFORM7 We substitute the above into equation 1 and take expectations over H t to get, DISPLAYFORM8 Doing the above replacements to upperbound the RHS of equation 2 and summing the inequation over t = 1 to t = T and taking the average and replacing the LHS by a lowerbound of it, we get, DISPLAYFORM9 Replacing into the RHS above the optimal choice of, DISPLAYFORM10 Thus stochastic RMSProp with the above step-length is guaranteed is reach ???criticality in number of iterations given by, T ??? DISPLAYFORM11 Lemma A.1.

At any time t, the following holds, DISPLAYFORM12 Proof.

DISPLAYFORM13 Now we introduce some new variables to make the analysis easier to present.

Let a pi := [???f p (x t )] i where p indexes the training data set, p ??? {1, . . . , k}. (conditioned on H t , a pi s are constants) This implies, DISPLAYFORM14 where the expectation is taken over the oracle call at the t th update step.

Further our instantiation of the oracle is equivalent to doing the uniformly at random sampling, (g t ) i ??? {a pi } p=1,...,k .Given that we have, DISPLAYFORM15 i +di where we have defined DISPLAYFORM16 This leads to an explicit form of the needed expectation over the t th ???oracle call as, DISPLAYFORM17 Substituting the above (and the definition of the constants a pi ) back into equation 3 we have, DISPLAYFORM18 Substituting this, the above expression can be written as, DISPLAYFORM19 Note that with this substitution, the RHS of the claimed lemma becomes, DISPLAYFORM20 Therefore our claim is proved if we show that for all i, DISPLAYFORM21 To further simplify, we define DISPLAYFORM22 ????? min .

We therefore need to show, DISPLAYFORM23 We first bound d i by recalling the definition of ?? f (from which it follows that a 2 pi ??? ?? 2 f ), DISPLAYFORM24 The inequality follows since ?? 2 ??? (0, 1]Putting this all together, we get, DISPLAYFORM25 Now our assumption that for all x, sign(???f p (x)) = sign(???f q (x)) for all p, q ??? {1, . . .

, k} leads to the conclusion that the term a pi a qi ??? 0.

And we had already shown in equation 5 that DISPLAYFORM26 Thus we have shown that (a i 1 k )(q i a i ) ??? 0 and this finishes the proof.

Proof.

By the L???smoothness condition and the update rule in Algorithm 2 we have, DISPLAYFORM0 For 0 < ?? DISPLAYFORM1 we now show a strictly positive lowerbound on the following function, DISPLAYFORM2 We define ?? t := max i=1,..,t ???f (x i ) and we solve the recursion for v t as, DISPLAYFORM3 .

This lets us write the following bounds, DISPLAYFORM4 Now we define, t := min k=1,..,t,i=1,..,d (???f (x k )) 2 i and this lets us get the following sequence of inequalities, DISPLAYFORM5 So combining equations 9 and 8 into equation 7 and from the exit line in the loop we are assured that ???f (x t ) 2 = 0 and combining these we have, DISPLAYFORM6 Now our definition of ?? 2 t allows us to define a parameter 0 < ?? t := DISPLAYFORM7 t and rewrite the above equation as, DISPLAYFORM8 We can as well satisfy the conditions needed on the variables, ?? t and ?? t by choosing, DISPLAYFORM9 Then the worst-case lowerbound in equation 10 becomes, DISPLAYFORM10 This now allows us to see that a constant step length ?? t = ?? > 0 can be defined as, DISPLAYFORM11 and this is such that the above equation can be written as, DISPLAYFORM12 2 .

This when substituted back into equation 6 we have, DISPLAYFORM13 This gives us, DISPLAYFORM14 Thus for any given > 0, T satisfying, DISPLAYFORM15 2 is a sufficient condition to ensure that the algorithm finds a point x result := arg min t=1,,.T ???f (x t ) 2 with ???f (x result ) 2 ??? 2 .Thus we have shown that using a constant step length of ?? = DISPLAYFORM16 Proof.

From the L???smoothness condition on f we have between consecutive iterates of the above algorithm, DISPLAYFORM17 Now the recursion for v t can be solved to get, DISPLAYFORM18 .

Substituting this in a lowerbound on the LHS of equation 13 we get, DISPLAYFORM19 Summing the above we get, DISPLAYFORM20 Now we substitute ?? t = ?? ??? t and invoke the definition of B and B u to write the first term on the RHS of equation 15 as, DISPLAYFORM21 Now we bound the second term in the RHS of equation 15 as follows.

Lets first define a function P (T ) as follows, DISPLAYFORM22 2 and that gives us, DISPLAYFORM23 So substituting the above two bounds back into the RHS of the above inequality 15and removing the factor of 1 ??? ?? T 2 < 1 from the numerator, we can define a point x result as follows, DISPLAYFORM24 Thus it follows that for T = O( 1 4 ) the algorithm 2 is guaranteed to have found at least one point DISPLAYFORM25 Proof.

Let us assume to the contrary that g t > for all t = 1, 2, 3. . .

..

We will show that this assumption will lead to a contradiction.

By L???smoothness of the objective we have the following relationship between the values at consecutive updates, DISPLAYFORM26 Substituting the update rule using a dummy step length ?? t > 0 we have, DISPLAYFORM27 The RHS in equation 16 above is a quadratic in ?? t with two roots: 0 and DISPLAYFORM28 So the quadratic's minimum value is at the midpoint of this interval, which gives us a candidate t th ???step length i.e ?? * DISPLAYFORM29 2 and the value of the quadratic at this point DISPLAYFORM30 That is with step lengths being this ?? * t we have the following guarantee of decrease of function value between consecutive steps, DISPLAYFORM31 Now we separately lower bound the numerator and upper bound the denominator of the RHS above.

DISPLAYFORM32 Further we note that the recursion of v t can be solved as, DISPLAYFORM33 k .

Now we define, t := min k=1,..,t,i=1,..,d (g 2 k ) i and this gives us, DISPLAYFORM34 We solve the recursion for m t to get, DISPLAYFORM35 Then by triangle inequality and defining ?? t := max i=1,..,t ???f (x i ) we have, m t ??? (1 ??? ?? t 1 )?? t .

Thus combining this estimate of m t with equation 18 we have, DISPLAYFORM36 m t To analyze this we define the following sequence of functions for each i = 0, 1, 2.., t DISPLAYFORM37 This gives us the following on substituting the update rule for m t , DISPLAYFORM38 Lets define, ?? t???1 := max i=1,..

,t???1 ???f (x i ) and this gives us for i ??? {1, .., t ??? 1}, DISPLAYFORM39 We note the following identity, DISPLAYFORM40 Now we use the lowerbounds proven on Q i ??? ?? 1 Q i???1 for i ??? {1, .., t ??? 1} and Q t ??? ?? 1 Q t???1 to lowerbound the above sum as, DISPLAYFORM41 We can evaluate the following lowerbound, DISPLAYFORM42 Next we remember that the recursion of v t can be solved as, v t = (1 ??? ?? 2 ) t k=1 ?? t???k 2 g 2 k and we define, ?? t := max i=1,..,t ???f (x i ) to get, DISPLAYFORM43 Now we combine the above and equation 18 and the known value of Q 0 = 0 (from definition and initial conditions) to get from the equation 20, DISPLAYFORM44 In the above inequalities we have set t = 0 and we have set, ?? t = ?? t???1 = ??.

Now we examine the following part of the lowerbound proven above, DISPLAYFORM45 Now we remember the assumption that we are working under i.e g t > .

Also by definition 0 < ?? 1 < 1 and hence we have 0 < ?? 1 ??? ?? t 1 < ?? 1 .

This implies, DISPLAYFORM46 > 1 where the last inequality follows because of our choice of as stated in the theorem statement.

This allows us to define a constant, DISPLAYFORM47 Similarly our definition of ?? allows us to define a constant ?? 2 > 0 to get, DISPLAYFORM48 Putting the above back into the lowerbound for Q t in equation 22 we have, DISPLAYFORM49 Now we substitute the above and equation 19 into equation 17 to get, DISPLAYFORM50 In the theorem statement we choose to call as the final ?? t the lowerbound proven above.

We check below that this smaller value of ?? t still guarantees a decrease in the function value that is sufficient for the statement of the theorem to hold.

A consistency check!

Let us substitute the above final value of the step length DISPLAYFORM51 DISPLAYFORM52 The RHS above can be simplified to be shown to be equal to the RHS in equation 24 at the same values of ?? 1 and ?? 2 as used above.

And we remember that the bound on the running time was derived from this equation 24.

Here we describe how we tune the hyper-parameters of each optimization algorithm.

NAG has two hyper-parameters, the step size ?? and the momentum ??.

The main hyper-parameters for RMSProp are the step size ??, the decay parameter ?? 2 and the perturbation ??.

ADAM, in addition to the ones in RMSProp, also has a momentum parameter ?? 1 .

We vary the step-sizes of ADAM in the conventional way of ?? t = ?? 1 ??? ?? t 2 /(1 ??? ?? t 1 ).

For tuning the step size, we follow the same method used in Wilson et al. (2017) .

We start out with a logarithmically-spaced grid of five step sizes.

If the best performing parameter was at one of the extremes of the grid, we tried new grid points so that the best performing parameters were at one of the middle points in the grid.

While it is computationally infeasible even with substantial resources to follow a similarly rigorous tuning process for all other hyper-parameters, we do tune over them somewhat as described below.

NAG The initial set of step sizes used for NAG were: {3e???3, 1e???3, 3e???4, 1e???4, 3e???5}. We tune the momentum parameter over values ?? ??? {0.9, 0.99}.

The initial set of step sizes used were: {3e???4, 1e???4, 3e???5, 1e???5, 3e???6}. We tune over ?? 2 ??? {0.9, 0.99}. We set the perturbation value ?? = 10 ???10 , following the default values in TensorFlow, except for the experiments in Section 5.1.

In Section 5.1, we show the effect on convergence and generalization properties of ADAM and RMSProp when changing this parameter ??.

Note that ADAM and RMSProp uses an accumulator for keeping track of decayed squared gradient v t .

For ADAM this is recommended to be initialized at v 0 = 0.

However, we found in the TensorFlow implementation of RMSProp that it sets v 0 = 1 d .

Instead of using this version of the algorithm, we used a modified version where we set v 0 = 0.

We typically found setting v 0 = 0 to lead to faster convergence in our experiments.

ADAM The initial set of step sizes used were: {3e???4, 1e???4, 3e???5, 1e???5, 3e???6}. For ADAM, we tune over ?? 1 values of {0.9, 0.99}. For ADAM, We set ?? 2 = 0.999 for all our experiments as is set as the default in TensorFlow.

Unless otherwise specified we use for the perturbation value ?? = 10 ???8 for ADAM, following the default values in TensorFlow.

Contrary to what is the often used values of ?? 1 for ADAM (usually set to 0.9), we found that we often got better results on the autoencoder problem when setting ?? 1 = 0.99.

In Figure 5 , we show the same effect of changing ?? as in Section 5.1 on a 1 hidden layer network of 1000 nodes, while keeping all other hyper-parameters fixed (such as learning rate, ?? 1 , ?? 2 ).

These other hyper-parameter values were fixed at the best values of these parameters for the default values of ??, i.e., ?? = 10 ???10 for RMSProp and ?? = 10 ???8 for ADAM.

To test whether our conclusions are consistent across different input dimensions, we do two experiments where we resize the 22 ?? 22 MNIST image to 17 ?? 17 and to 12 ?? 12.

Resizing is done using TensorFlow's tf.image.resize images method, which uses bilinear interpolation.

17 ?? 17 Figure 9 shows results on input images of size 17 ?? 17 on a 3 layer network with 1000 hidden nodes in each layer.

Our main results extend to this input dimension, where we see ADAM with ?? 1 = 0.99 both converging the fastest as well as generalizing the best, while NAG does better than ADAM with ?? 1 = 0.9.

In FIG1 , we present results on additional neural net architectures on mini-batches of size 100 with an input dimension of 22 ?? 22.

We see that most of our full-batch results extend to the minibatch case.(a) 1 hidden layer; 1000 nodes (b) 3 hidden layers; 1000 nodes each (c) 9 hidden layers; 1000 nodes each (d) 1 hidden layer; 1000 nodes (e) 3 hidden layers; 1000 nodes each (f) 9 hidden layers; 1000 nodes each (g) 1 hidden layer; 1000 nodes (h) 3 hidden layers; 1000 nodes each (i) 9 hidden layers; 1000 nodes each FIG1 : Experiments on various networks with mini-batch size 100 on full MNIST dataset with input image size 22 ?? 22.

First row shows the loss on the full training set, middle row shows the loss on the test set, and bottom row shows the norm of the gradient on the training set.

<|TLDR|>

@highlight

In this paper we prove convergence to criticality of (stochastic and deterministic) RMSProp and deterministic ADAM for smooth non-convex objectives and we demonstrate an interesting beta_1 sensitivity for ADAM on autoencoders. 

@highlight

This paper presents a convergence analysis of RMSProp and ADAM in the case of smooth non-convex functions