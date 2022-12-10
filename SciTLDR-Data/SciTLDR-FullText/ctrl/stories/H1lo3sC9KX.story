Asynchronous distributed gradient descent algorithms for training of deep neural networks are usually considered as inefficient, mainly because of the Gradient delay problem.

In this paper, we propose a novel asynchronous distributed algorithm that tackles this limitation by well-thought-out averaging of model updates, computed by workers.

The algorithm allows computing gradients along the process of gradient merge, thus, reducing or even completely eliminating worker idle time due to communication overhead, which is a pitfall of existing asynchronous methods.

We provide theoretical analysis of the proposed asynchronous algorithm, and show its regret bounds.

According to our analysis, the crucial parameter for keeping high convergence rate is the maximal discrepancy between local parameter vectors of any pair of workers.

As long as it is kept relatively small, the convergence rate of the algorithm is shown to be the same as the one of a sequential online learning.

Furthermore, in our algorithm, this discrepancy is bounded by an expression that involves the staleness parameter of the algorithm, and is independent on the number of workers.

This is the main differentiator between our approach and other solutions, such as Elastic Asynchronous SGD or Downpour SGD, in which that maximal discrepancy is bounded by an expression that depends on the number of workers, due to gradient delay problem.

To demonstrate effectiveness of our approach, we conduct a series of experiments on image classification task on a cluster with 4 machines, equipped with a commodity communication switch and with a single GPU card per machine.

Our experiments show a linear scaling on 4-machine cluster without sacrificing the test accuracy, while eliminating almost completely worker idle time.

Since our method allows using commodity communication switch, it paves a way for large scale distributed training performed on commodity clusters.

Distributed training of deep learning models is devised to reduce training time of the models.

Synchronous distributed SGD methods, such as BID0 and BID7 , perform training using mini-batch size of several dozens of thousands of images.

However, they either require expensive communication switch for fast gradient sharing between workers or, otherwise, introduce a high communication overhead during gradient merge, where workers are idle waiting for communicating gradients over communication switch.

Distributed asynchronous SGD methods reduce the communication overhead on one hand, but usually introduce gradient delay problem on the other hand, as described in BID0 .

Indeed, usually in an asynchronous distributed approach, a worker w obtains a copy of the central model, computes a gradient on this model and merges this gradient back into the central model.

Note, however, that since the worker obtained the copy of the central model till it merges its gradient back into the central model, other workers could have merged their gradients into the central model.

Thus, when the worker w merges its gradient into a central model, that model may have been updated and, thus, the gradient of the worker w is delayed, leading to gradient delay problem.

We will refer to algorithms that suffer from gradient delay problem as gradient delay algorithms, e.g. Downpour SGD BID1 .As our analysis reveals in Section 3, the quantity that controls the convergence rate of an asynchronous distributed algorithm, is maximal pairwise distance -the maximal distance between local models of any pair of workers at any iteration.

Usually gradient delay algorithms do not limit this distance and it may depend on the number of asynchronous workers, which may be large in large clusters.

This may explain their poor scalability, convergence rate and struggle to reach as high test accuracy as in synchronous SGD algorithms, as experimentally shown in BID0 .While Elastic Averaging SGD Zhang et al. (2015) is also a gradient delay algorithm, it introduces a penalty for workers, whose models diverge too far from the central model.

This, in turn, helps to reduce the maximal pairwise distance between local models of workers and, thus, leads to better scalability and convergence rate.

In contrast, our analysis introduces staleness parameter that directly controls the maximal pair distance of the asynchronous workers.

Our analysis builds on the work of BID10 , who studied convergence rate of gradient delay algorithms, when the maximum delay is bounded.

They provided analysis for Lipschitz continuous losses, strongly convex and smooth losses.

While they show that bounding staleness can improve convergence rate of a gradient delay algorithm, in their algorithm each worker computes exactly one gradient and is idle, waiting to merge the gradient with PS model and download the updated PS model back to the worker.

The main contributions of this paper are the following.

We present and analyze a new asynchronous distributed SGD method that both reduces idle time and eliminates gradient delay problem.

Our main theoretic result shows that an asynchronous distributed SGD algorithm can achieve convergence rate as good as in sequential online learning.

We support this theoretic result by conducting experiments that show that on a cluster with up to 4 machines, with a single GPU per machine and a commodity communication switch, our asynchronous method achieves linear scalability without degradation of the test accuracy.

Below we describe two algorithms.

We will use the terms model and parameter vector to refer to the collection of trainable parameters of the model.

Each worker in these algorithms starts computing gradients from a copy of a PS model and, prior to merging with the central model at PS, the worker can compute either a single gradient or several gradients, advancing the local model using all these gradients.

In the sequel we will use term model update of a worker -the difference between the last local model prior to the merge and the latest copy of PS model, from which the worker started computing gradients.

The key idea of our approach is to reduce gradient delay, as described in Algorithm 1.

To achieve this goal, the merge process waits till each worker has at least one gradient computed locally.

Then, model updates from all the workers are collected and averaged and the average model update is used to update the model at PS.

This is the synchronous part of our hybrid algorithm.

In this way we replace gradient delay by averaging gradients, which is widely used as a technique to increase mini-batch size.

Workers are allowed to compute gradients asynchronously to each other and to the merge process to reduce wait times, when workers are idle.

This is the asynchronous part of our hybrid algorithm.

Furthermore, the idea of computing several gradients to form a model update is used to hide communication overhead of gradient merge with useful computation of gradients at the workers.

Master starts with computing the initial version of the model in line 27 and provides it for transferring to workers.

Then Master waits till each worker has computed at least one gradient and used it to advance the worker's local model.

When this happens, Master instructs on transferring a model update w.∆x from each worker to PS, where model updates from all workers are averaged and the average is used to advance the PS model.

Finally the updated PS model is provided to all workers.

A worker starts with assigning value 0 to each variable, except variable staleness s, which is assigned the maximal staleness τ .

In line 5 the worker checks if the staleness s has already achieved its maximal value and in this case it waits in line 6 till an initial version of the PS model is transferred to the worker.

When the PS model is transferred to the worker, in lines 9 and 10, the worker initializes the two model variables x and x init with the PS model, since at this stage model update w.∆x is still 0.

Next, the worker sets staleness to 0.

Lines 13-15 comprise a usual update of the local model.

In line 16, the worker notifies Master on availability of a non-zero model update w.∆x.

Now in line 17, the worker advances the local iteration counter i+ = 1.

It also advances staleness s, since with advancing the local model with one gradient, the local model gets far away from the latest version of PS model, stored in line 9 in x init , by one more gradient.

In this way, the worker can perform several iterations, computing a gradient and advancing the local model in lines 13-15, as long as the current staleness s does not hit the maximal staleness τ .Assume now that at some point in time, Master requests to transfer model update w.∆x to PS in line 30.

Each worker receives this request in line 21 of Thread 2.

In this case, the worker releases the model update in line 22 to transfer to PS to merge with the PS model and sets the local model x to x init , to indicate that the model update is re-initialized to 0.

The value i init is re-initialized to i to indicate that the number of gradients, accumulated in model update x − x init is 0.Algorithm 1: Asynchronous SGD without Gradient Delay.

To simplify notation, we denote β the cycle length, i.e. β = τ 2 .

At the end of each cycle, e.g. at iteration jβ , all workers synchronize with PS: each worker acquires the copy of a new PS model and provides its accumulated model update to PS.

If PS does not have enough time to merge model updates from the previous cycle and transfer the new model to the workers, workers wait for the new PS model in line 6.

When a worker receives the new PS model, it computes the model update w.∆x in line 7 and provides it to PS in line 8.

Note that due to the cycle length of β iterations, model update w.∆x is computed using β locally computed gradients.

Now, in line 9, the worker advances the new PS model using the model update and sets the resulting model to x init and x. Note that at this point in time, at each worker, the parameter vector x is sum of the new PS model, which is the same in all the workers, and the model update w.∆x that is computed using β gradients, computed locally in the worker.

This means that the worker can perform β additional iterations before hitting the staleness boundary of τ .

Now, after the synchronization with PS, each worker computes β gradients locally and uses them to advance the local model in lines 11-13.

At this point in time, each worker hits the staleness boundary of τ .

In parallel with this computation of model update in each worker, in lines 20-24, Master transfers model updates that workers provided to PS in iteration jβ, merges them with PS model and provides the new model to the workers.

When each worker finishes iteration (j + 1)β, the current cycle completes and, again, synchronization starts between workers and PS.Note that workers in Algorithm 2 may be idle, waiting for the new PS model in line 6, only if the communication between workers and PS is slow and does not allow transferring model updates from workers to PS, merge them with the PS model and transfer the new PS model back to the workers before workers complete computing β gradients.

Also note that further increasing staleness, reduces or completely eliminates worker idle time.

In this section, we analyze Algorithm 2.

In the analysis we adopt the notation of BID10 .

In the proof of Lemma 3.4 and of Theorem 3.5, we use derivation similar to BID10 , while adding terms, which are relevant for our specific algorithms.

The most of the analysis is our own contribution.

We will point out to parts that we borrow from prior research work.

Our main result is Theorem 3.9, where we show that the convergence rate of Algorithm 2 is O(τ 2 + √ T ).

The algorithm has two phases.

In the first phase, during the early iterations, the asynchronous training leads to a slow-down, expressed in the term τ 2 .

However, in the second phase, during later iterations, the asynchronous training appears to be harmless.

We show that, if T τ , the convergence rate of Algorithm 2 is as good as in sequential online learning.

Note that the convergence rate, stated in Theorem 3.9, assumes that the standard deviation of gradient computation is inversely proportional to the maximal staleness τ .

In practice, this means that for each given value of maximal staleness τ , one needs to set the size of mini-batch at each worker so that the standard deviation of gradient computation gets below Denote convex (cost) functions by f i : X → R, and their parameter vector by x. Our goal is to find a sequence of x i such that the cumulative loss i f i (x i ) is minimized.

With some abuse of notation, we identify the average empirical and expected loss both by f * .

This is possible, simply by redefining p(f ) to be the uniform distribution over F .

Denote by DISPLAYFORM0 the average risk.

We assume that x * exists (convexity does not guarantee a bounded minimizer) and that it satisfies ||x * || ≤ R (this is always achievable, simply by intersecting X with the unit-ball of radius R).We remind that in Algorithm 2, in each cycle each worker computes an update to its local parameter vector, based on β locally computed gradients and PS averages these per-worker updates from all the workers to update its own parameter vector.

In Algorithm 2 at time t, each worker w computes the gradient of the same function f on its own parameter vector x t,w and its own mini-batch.

We denote this function at worker w and time t f t,w and denote the gradient of this function, computed at the local parameter vector x t,w , g t,w = ∇f t,w (x t,w ).For the analysis, we define the global parameter vector at time t (as opposite to per-worker parameter vector x t,w ) as average of per-worker parameter vectors DISPLAYFORM1 Also we denote DISPLAYFORM2 We assume that each f t,w , and thus f t , is convex, and subdifferentials of f t are bounded ||∇f t (x)|| ≤ L by some L > 0.

Denote by x * the minimizer of f * (x).

We want to find a bound on the regret R, associated with a sequence X = x 1 , ..., x T of parameter vectors DISPLAYFORM3 Such bounds can be converted into bounds on the expected loss, as in BID4 , for an example.

Note that with the definitions (1) and (2), the regret in (3) is well-defined.

We denoteg DISPLAYFORM4 This is how a gradient is computed in synchronous distributed SGD algorithms.

Since all f t are convex, we can upper bound DISPLAYFORM5 Let us define a distance function between x and x : D(x||x ) = 1 2 ||x − x || 2 .

In Algorithm 2, each worker at the beginning of cycle i, i.e. at time iβ, receives the copy of a new PS parameter vector.

In Lemmas 3.1 -3.3 we study properties of Algorithm 2 at this point in time.

In Lemma 3.1, we show that the copy of PS model at time iβ, equals to the average of local parameter vectors of workers from iteration (i − 1)β, which, according to (1), equals to x (i−1)β .Lemma 3.1.

The copy of a new PS parameter vector that each worker receives at iteration (i + 1)β, equals to the average of local parameter vectors of workers from iteration iβ DISPLAYFORM6 The proof may be found in Appendix A. Next, when a worker receives at iteration iβ the copy of a new PS parameter vector, which, according to Lemma 3.1, equals to x (i−1)β , it adds to this parameter vector its latest model update x w,iβ ← x (i−1)β + ∆w.x.

Note that this operation resets all the local parameter vectors of the worker, computed in the last cycle: the operation moves all the local parameter vectors {x w,(i−1)β+t |t = 0, . . .

, β} along the vector x w,(i−1)β − x (i−1)β towards the average parameter vector x (i−1)β : DISPLAYFORM7 Lemma 3.2 shows that computing average parameter vector (1) is invariant under rest operation (7).

Lemma 3.2.

Computing average parameter vector, according to (1), is invariant to the reset operation (7), i.e. for each i ∈ N DISPLAYFORM8 The proof may be found in Appendix A. Now, Lemma 3.3 bounds the distance between parameter vectors for any pair of workers after the reset operation at iteration iβ.

Lemma 3.3.

Suppose gradients of cost functions f t are bounded ||∇f t (x)|| ≤ L by some L. Let w, w ∈ W be any two workers.

Then after the reset operation (7), at iteration iβ, DISPLAYFORM9 The proof may be found in Appendix A. After the reset operation at iteration iβ, in the next β iterations, each worker computes β gradients locally to advance its local parameter vector DISPLAYFORM10 Using this expression, we can re-write (1) DISPLAYFORM11 We abuse notation and denote an average over gradients g iτ +t,w as DISPLAYFORM12 Note the difference between (4) and (11).

We use expression (4) as one way to define the average gradient, with which we start to bound regret in (5).

In this expression, each gradient uses the same parameter vector, defined in (1), to compute its gradient.

Expression (11), is another way to compute an average gradient, where each worker uses its own local parameter vector to compute gradient.

With notation (11) we rewrite (10) DISPLAYFORM13 Next, to prove our regret bounds, we adapt Lemma 1 from BID10 .

In BID10 , Lemma 1 is proved for an asynchronous algorithm, in which each worker computes exactly one gradient and transfers it to PS to merge with its parameter vector.

We adapt this lemma to Algorithm 2, in which each worker computes β gradients locally and then all the workers transfer their updates of their local parameter vectors to PS, which averages them and uses this average to update its own parameter vector.

Lemma 3.4.

For all x * , for all i and 0 ≤ t < β, if X = R n , the following expansion holds: DISPLAYFORM14 The proof may be found in Appendix A. The decomposition (13) is very similar to standard regret decomposition bounds, such as BID9 .

We add to this decomposition a new term < x iβ+t − x * ,g iβ+t − g iβ+t > to adapt the analysis to peculiarities of Algorithm 2.

This term characterizes the difference between two ways to compute an average gradient at a specific time.

In Algorithm 2, at iteration iβ, for each i, each worker starts computing gradients after it adds its local update of its local parameter vector to the copy of a common PS parameter vector and computes β gradients locally.

As workers compute more gradients in iterations iβ + t, t = 1, . . . , β, their local parameter vectors get more far apart from each other.

However, for sufficiently small step size, the distance between local parameter vectors in different workers remains small.

The key to proving our bounds is to impose further smoothness constraints on f t .

The rationale is quite simple: we want to ensure that small changes in x do not lead to large changes in the gradient.

More specifically we assume that the gradient of f t is a Lipschitz-continuous function.

That is, DISPLAYFORM15 for some constant H. Following Theorem 2 from BID10 , we use Lemma 3.4 to prove Theorem 3.5.

The key difference between Theorem 3.5 and Theorem 2 from BID10 , is that we introduce a new expression and bound it to adapt the analysis to Algorithm 2.

Theorem 3.5.

Suppose gradients of cost functions f t are bounded ||∇f t (x)|| ≤ L by some L and that H also upper-bounds the change in the gradients, as in FORMULA1 .

Also suppose that DISPLAYFORM16 , for some constant σ > 0, the regret of Algorithm 2 is bounded by DISPLAYFORM17 Consequently, for DISPLAYFORM18 The proof may be found in Appendix A. According to Theorem 3.5, the convergence rate of Algorithm 2 is O(τ 2 + √ τ T ).

As in BID9 , we note that this result is expected, because an adversary can rearrange training instances so that each worker receives training instances that are very similar to training instances in other workers during τ asynchronous iterations.

In this case multiple workers do not perform better than a single worker and, thus, parallel algorithm is not better than a sequential one.

We will use the results of Theorem 3.5 to prove our main result in Theorem 3.9 3.1 DECORRELATED GRADIENT ANALYSIS In this section we assume that training samples at different workers are drawn independently from the same underlying distribution.

Also, to improve the convergence rate, we should limit the divergence of different workers from each other, as they asynchronously advance their local parameter vectors.

Namely, we introduce an assumption on variance of gradients, computed in the same point at different workers.

We denote g * = ∇f * and assume that variation of gradients at different workers, is modeled by an additive Gaussian noise ∇f t,w (x) = g * (x) + e t,w , for e t,w ∼ N (0, C), where C is a covariance matrix.

We start with the following lemma that bounds the distance between parameter vectors at any two workers w, w ∈ W after each one of them makes j asynchronous steps, starting from parameter vectors x iβ,w and x iβ,w .

Lemma 3.6.

Assume that variation of the gradient of the cost function is governed by an additive Gaussian noise ∇f t,w (x) = g * (x) + e t,w , (17) for e t,w ∼ N (0, C) for covariance matrix C. Then for any two worker w, w ∈ W , DISPLAYFORM19 The proof may be found in Appendix A. Lemma 3.6 bounds the distance between local parameter vectors in two different workers, as they proceed with asynchronous iterations.

Lemma 3.7 uses this bound to prove a bound on the expected value of this difference.

Lemma 3.7.

Denote var(|e t,w |) = s. Also, in addition to the conditions of Lemma 3.6, assume that the cost functions f t are i.i.d.

Then, for DISPLAYFORM20 the expectation of the difference ||x iτ +j,w − x iτ +j,w || is bounded by DISPLAYFORM21 The proof may be found in Appendix A. Using (20) for j = β, before reset operation FORMULA7 , DISPLAYFORM22 After reset operation, updates of local parameter vectors of w and w , computed in iterations t = iβ + 1, . . .

, (i + 1)β, are added to the common copy of a PS model.

This results in reduction of RHS of FORMULA1 by ||x iβ,w − x iβ,w ||, resulting in DISPLAYFORM23 In FORMULA2 , the bound on expected distance between local parameter vectors of workers w and w at iteration (i + 1)β, depends on the distance at the previous cycle, at iteration iβ.

Lemma 3.8 develops a similar bound without the dependence on the distance from the previous cycle.

Lemma 3.8.

Let i 0 be the smallest index, s.t.

i 0 β ≥ t 0 , for t 0 defined in (19).

Then, for DISPLAYFORM24 and any i ≥ i 0 + j 0 , DISPLAYFORM25 The proof may be found in Appendix A. Now, when we bounded the difference between parameter vectors in different workers, we can use this bound along with the assumption (14) to bound the difference between two ways to compute an average gradient: (4) and (11).

This leads to our main result -bound on the expected regret of Algorithm 2.

Theorem 3.9.

In addition to the conditions of Theorem 3.5, assume that the cost functions f t are i.i.d.

and that variation of gradients of the cost functions is governed by an additive Gaussian law, i.e. ∇f t,w (x) = g * (x) + e t,w , for e t,w ∼ N (0, C), s.t.

DISPLAYFORM26 DISPLAYFORM27 for some constant σ > 0, the expected regret of Algorithm 2 is bounded as follows DISPLAYFORM28 DISPLAYFORM29 The proof may be found in Appendix A.

In this section we provide an initial experimental support for the effectiveness of our algorithm and show discrepancy in local parameter vectors of the asynchronous workers in two approaches: averaging models updates and gradient delay.

We start with the discrepancy.

For this experiment we trained ResNet50 model BID6 ) on CIFAR-10 data (Krizhevsky et al.) .

We show the results in FIG2 , where different gradient average plots correspond to different values of staleness.

To produce a point in a gradient average plot for a given values of staleness and the number of workers, each worker starts computing an update to its local parameter vector from a parameter vector that is common to all the workers.

Workers proceed computing the gradients, updating their local parameter vectors asynchronously until each worker computes the number of gradients, as the chosen value of staleness.

At this point we compute the average over the last parameter vectors in the workers.

Next we compute the average distance between the last parameter vectors of the workers and the average parameter vector.

This average distance is plotted in FIG2 for the corresponding values of staleness and the number of workers.

To produce a point in the gradient delay plot, we simulate the worst case in gradient delay: workers read the central parameter vector P S.x in a sequence.

The first worker reads the initial value of P S.x, the second after P S.x is advanced with one gradient, the third after P S.x is advanced with two gradients, etc., until the last worker read P S.x after it is advanced the number of times as the number workers minus 1.

Each worker computes only one gradient and uses it to advance its local parameter vector.

Since the last value of P S.x represents the most up-to-date model in the system, the average distance is now computed between this model and all the local models of the workers.

This average distance is shown in the gradient delay plot.

As we see, from FIG2 , staleness is an effective tool to keep low discrepancy of local parameter vectors of asynchronous workers -the discrepancy is roughly linear in the value of the maximal staleness parameter.

In contrast, in gradient delay, this discrepancy is linear in the number of workers and, thus, is limited only by the number of workers, which may be large in very large clusters.

Next, we provide an initial experimental support for scalability of our algorithm.

We trained GoogleNet model BID5 ) on ImageNet ILSVRC12 dataset BID3 ) on a cluster with 4 machines.

Each machine was equipped with a single Nvidia GeForce GTX TITAN X GPU card.

The machines were interconnected using 1 giga bit communication switch.

We used mini-batches of size 32 images and set the maximal staleness parameter to the value of 16.

Also we implemented a linear scaling rule, where training with n workers, we increase the learning rate n times and reduce the number of iterations in each worker n times.

We added a linear warm-up of 50K iterations to gradually increase the base learning rate from 0.01 to n · 0.01.

After training, we test the resulting model using ImageNets own 50,000 images validation set.

From FIG3 and Figure 3 , we see that our method achieves linear scalability without degradation of the test accuracy.

We measured average idle time of workers and found it to be practically 0 for staleness value of at least 8.

We presented a new asynchronous distributed SGD method.

We show empirically that it reduces both idle time and gradient delay.

We analyze the synchronous part of the algorithm, and show theoretical regret bounds.

The proposed method shows promising results on distributed training of deep neural networks.

We show that our method eliminates waiting times, which allows significant improvements in run time, compared to fully synchronous setup.

The very fact of efficient hiding of communication overhead opens opportunity for distributed training over commodity clusters.

Furthermore, the experiments show linear scaling of training time from 1 to 4 GPU's without compromising the final test accuracy.

Proof.

We start with RHS of (8) DISPLAYFORM0 By the definition of average parameter vector (1), the first summand in RHS of (29) equals x iβ+t , while the last two summands cancel each other.

Proof of Lemma 3.3.Proof.

At iteration (i − 1)β after reset operation FORMULA7 , workers w and w start updating their local parameter vectors for β iterations, so that at the end of the cycle before the reset operation (7), DISPLAYFORM1 The reset operation FORMULA7 at iteration iβ adds the updates of workers w and w , computed at iterations (i − 1)β + 1, . . .

, iβ, to the common copy of PS model, so that x (i−1)β,w − x (i−1)β,w = 0.

Thus, after the reset operation DISPLAYFORM2 Finally, from (31), our assumption on the size of the gradient ∇f t (x) and from decreasing learning rate, during this cycle that started at iteration (i − 1)β, the distance between workers w and w grows at most by DISPLAYFORM3 Proof of Lemma 3.4.Proof.

We decompose our progress as follows DISPLAYFORM4 To prove (33) from (32), we used (12) Dividing both sides by η iτ +t and moving < x iτ +t − x * ,g iτ +t > to the LHS completes the proof.

Proof of Theorem 3.5.Proof.

First we state a useful inequality: For n vectors a i , i = 1, . . .

, n (by induction on n): DISPLAYFORM5 Also we will use the following sum bounds: DISPLAYFORM6 and DISPLAYFORM7 We start with summing (13) along iterations: DISPLAYFORM8 Next, we borrow the derivation of expressions (38) and (39) from the proof of Theorem 2, BID9 .

Note, however, that we added a new term to (37) -the last term that is specific to Algorithm 2.

This term does not appear in the proof of Theorem 2, BID9 .

By the Lipschitz property of gradients and the definition of η t , we can bound the first summand of the above regret expression via DISPLAYFORM9 Also DISPLAYFORM10 We omit the negative factor − DISPLAYFORM11 .

Now we start the analysis of the last term in (37), which is new and specific to Algorithm 2.

Using (34), we bound the last summand of (37) DISPLAYFORM12 Substituting FORMULA3 , FORMULA3 and FORMULA4 into FORMULA3 , we get DISPLAYFORM13 We use (14) in the above expression DISPLAYFORM14 Now, note that the last summand in (50) corresponds to the summand in (18) for k = j − 1.

This completes the proof.

Proof of Lemma 3.7.Proof.

First note that from (19) it follows that η t βH ≤ 0.1 .Since η i shrinks down, as i grows up, and η t H + 1 > 1, (18) yields ||x iβ+j,w − x iβ+j,w || ≤ (η iβ H + 1) β · ||x iβ,w − x iβ,w || + η iβ β−1 k=0 ||e iβ+k,w − e iβ+k,w || .After the expansion of (η iβ H + 1) β in (52) into Taylor sequence and applying a simple algebra DISPLAYFORM15 From FORMULA1 we can bound DISPLAYFORM16 Assigning FORMULA4 into FORMULA2 , ||x iβ+j,w − x iβ+j,w || ≤ 1.2 · ||x iβ,w − x iβ,w || + 1.2 · η iβ β−1 k=0 ||e iβ+k,w − e iβ+k,w || .Since e iτ +k,w and e iτ +k,w are i.i.d.

with mean 0, E||e iβ+k,w −

e iβ+k,w || = var(e iβ+k,w − e iβ+k,w ) = 2var(e iβ+k,w ) .Taking expectation of the both sides of (55), substituting (56) into the resulting inequality and using the assumption on variance of random variables e t,w , we complete the proof.

Proof of Lemma 3.8.Proof.

From Lemma 3.3, ||x i0β,w − x i0β,w || ≤ 2Lτ η (i0−1)β .From FORMULA2 , each cycle j after iteration i 0 β, reduces the distance between w and w by factor of 0.2, while adding to the distance the value of 1.2 · η iβ+j τ s. This means that after the number of cycles j 0 j 0 = log 1.2 · η i0β τ s 2Lτ η (i0−1)β = log s L , for any i ≥ i 0 + j 0 , the first summand in (22) gets bounded by 0.8 · η iβ τ s.

Proof of Theorem 3.9.Proof.

To prove this theorem, we follow the proof of Theorem 3.5 and develop an alternative bound on ||x t,w − x t || in (42).We will split the sum in (40) into two ranges of t: t < t 0 + j 0 β and t ≥ t 0 + j 0 β for t 0 , j 0 , defined in FORMULA1 and FORMULA2 respectively.

To simplify notation, from (23), we can assume that j 0 is a constant and we can assume that t 0 + j 0 ≤ 2t 0 .

For t < 2t 0 , we use (48) to show 2t0−1 t=1 ||x t − x * || · ||g t −g t || ≤ 2τ F LH(2τ + 2σ DISPLAYFORM17 For t ≥ 2t 0 , we first observe DISPLAYFORM18 Next we use (24) to bound the above expression.

E||x t,w − x t || ≤ 2η t−τ τ s .Substituting this bound into (42), we get E||g t −g t || ≤ 2Hη t−τ τ s .Using FORMULA1 Combining FORMULA5 and FORMULA2 in FORMULA4 and using the definition (19) of t 0 DISPLAYFORM19 Using the assumption (25), and substituting into (41) we prove (26).

Finally we use value σ = F L to prove (27).

<|TLDR|>

@highlight

A method for an efficient asynchronous distributed training of deep learning models along with theoretical regret bounds.

@highlight

The paper proposes an algorithm to restrict the staleness in asynchronous SGD and provides theoretical analysis

@highlight

Proposes a hybrid-algorithm to eliminate the gradient delay of asynchronous methods.