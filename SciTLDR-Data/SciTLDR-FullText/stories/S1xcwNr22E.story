We analyze the dynamics of training deep ReLU networks and their implications on generalization capability.

Using a teacher-student setting, we discovered a novel relationship between the gradient received by hidden student nodes and the activations of teacher nodes for deep ReLU networks.

With this relationship and the assumption of small overlapping teacher node activations, we prove that (1) student nodes whose weights are initialized to be close to teacher nodes converge to them at a faster rate, and (2) in over-parameterized regimes and 2-layer case, while a small set of lucky nodes do converge to the teacher nodes, the fan-out weights of other nodes converge to zero.

This framework provides insight into multiple puzzling phenomena in deep learning like over-parameterization, implicit regularization, lottery tickets, etc.

We verify our assumption by showing that the majority of BatchNorm biases of pre-trained VGG11/16 models are negative.

Experiments on (1) random deep teacher networks with Gaussian inputs, (2) teacher network pre-trained on CIFAR-10 and (3) extensive ablation studies validate our multiple theoretical predictions.

Although neural networks have made strong empirical progress in a diverse set of domains (e.g., computer vision (16; 32; 10), speech recognition (11; 1), natural language processing (22; 3), and games (30; 31; 35; 23)), a number of fundamental questions still remain unsolved.

How can Stochastic Gradient Descent (SGD) find good solutions to a complicated non-convex optimization problem?

Why do neural networks generalize?

How can networks trained with SGD fit both random noise and structured data (38; 17; 24), but prioritize structured models, even in the presence of massive noise (27)?

Why are flat minima related to good generalization?

Why does overparameterization lead to better generalization (25; 39; 33; 26; 19)?

Why do lottery tickets exist (6; 7)?In this paper, we propose a theoretical framework for multilayered ReLU networks.

Based on this framework, we try to explain these puzzling empirical phenomena with a unified view.

We adopt a teacher-student setting where the label provided to an over-parameterized deep student ReLU network is the output of a fixed teacher ReLU network of the same depth and unknown weights ( FIG0 ).

In this perspective, hidden student nodes are randomly initialized with different activation regions. (Fig. 2(a) ).

During optimization, student nodes compete with each other to explain teacher nodes.

Theorem 4 shows that lucky student nodes which have greater overlap with teacher nodes converge to those teacher nodes at a fast rate, resulting in winner-takeall behavior.

Furthermore, Theorem 5 shows that if a subset of student nodes are close to the teacher nodes, they converge to them and the fan-out weights of other irrelevant nodes of the same layer vanishes.

With this framework, we can explain various neural network behaviors as follows:Fitting both structured and random data.

Under gradient descent dynamics, some student nodes, which happen to overlap substantially with teacher nodes, will move into the teacher node and cover them.

This is true for both structured data that corresponds to small teacher networks with few intermediate nodes, or noisy/random data that correspond to large teachers with many intermediate nodes.

This explains why the same network can fit both structured and random data ( Fig. 2(a-b) ).Over-parameterization.

In over-parameterization, lots of student nodes are initialized randomly at each layer.

Any teacher node is more likely to have a substantial overlap with some student nodes, which leads to fast convergence ( Fig. 2(a) and (c), Thm.

4), consistent with (6; 7).

This also explains that training models whose capacity just fit the data (or teacher) yields worse performance (19).Flat minima.

Deep networks often converge to "flat minima" whose Hessian has a lot of small eigenvalues (28; 29; 21; 2).

Furthermore, while controversial (4), flat minima seem to be associated with good generalization, while sharp minima often lead to poor generalization (12; 14; 36; 20).

In our theory, when fitting with structured data, only a few lucky student nodes converge to the teacher, while for other nodes, their fan-out weights shrink towards zero, making them (and their fan-in weights) irrelevant to the final outcome (Thm. 5), yielding flat minima in which movement along most dimensions ("unlucky nodes") results in minimal change in output.

On the other hand, sharp min- Figure 2 .

Explanation of implicit regularization.

Blue are activation regions of teacher nodes, while orange are students'.

(a) When the data labels are structured, the underlying teacher network is small and each layer has few nodes.

Over-parameterization (lots of red regions) covers them all.

Moreover, those student nodes that heavily overlap with the teacher nodes converge faster (Thm. 4), yield good generalization performance.

(b) If a dataset contains random labels, the underlying teacher network that can fit to it has a lot of nodes.

Over-parameterization can still handle them and achieves zero training error.(a) (b) (c) Figure 3 .

Explanation of lottery ticket phenomenon.

(a) A successful training with over-parameterization (2 filters in the teacher network and 4 filters in the student network).

Node j3 and j4 are lucky draws with strong overlap with two teacher node j ??? 1 and j ??? 2 , and thus converges with high weight magnitude.

(b) Lottery ticket phenomenon: initialize node j3 and j4 with the same initial weight, clamp the weight of j1 and j2 to zero, and retrain the model, the test performance becomes better since j3 and j4 still converge to their teacher node, respectively.

(c) If we reinitialize node j3 and j4, it is highly likely that they are not overlapping with teacher node j ima is related to noisy data ( Fig. 2(d) ), in which more student nodes match with the teacher.

Implicit regularization.

On the other hand, the snapping behavior enforces winner-take-all: after optimization, a teacher node is fully covered (explained) by a few student nodes, rather than splitting amongst student nodes due to over-parameterization.

This explains why the same network, once trained with structured data, can generalize to the test set.

Lottery Tickets.

Lottery Tickets (6; 7) is an interesting phenomenon: if we reset "salient weights" (trained weights with large magnitude) back to the values before optimization but after initialization, prune other weights (often > 90% of total weights) and retrain the model, the test performance is the same or better; if we reinitialize salient weights, the test performance is much worse.

In our theory, the salient weights are those lucky regions (E j3 and E j4 in Fig. 3 ) that happen to overlap with some teacher nodes after initialization and converge to them in optimization.

Therefore, if we reset their weights and prune others away, they can still converge to the same set of teacher nodes, and potentially achieve better performance due to less interference with other irrelevant nodes.

However, if we reinitialize them, they are likely to fall into unfavorable regions which cannot cover teacher nodes, and therefore lead to poor performance ( Fig. 3(c) ), just like in the case of under-parameterization.

The details of our proposed theory can be found in Appendix (Sec. 5).

Here we list the summary.

First we show that for multilayered ReLU, there exists a relationship between the gradient g j (x) of a student node j and teacher and student's activations of the same layer (Thm. 1): DISPLAYFORM0 (1) where f j ??? is the activation of node j??? in the teacher, and j is the node at the same layer in the student.

For each node j, we don't know which teacher node corresponds to it, hence the linear combination terms.

Typically the number of student nodes is much more than that of teachers'.

Thm.

1 applies to arbitrarily deep ReLU networks.

Then with a mild assumption (Assumption 1), we can write the gradient update rule of each layer l in the following DISPLAYFORM1 where L and L * are correlations matrix of activations from the bottom layers, and H and H * are modulation matrix from the top layers.

We then make an assumption that different teacher nodes of the same layer have small overlap in node activations (Assumption 3 and FIG4 , and verify it in VGG16/VGG11 by showing that the majority of their BatchNorm bias are negative FIG0 .

With this assumption, we prove two theorems:??? When the number of student nodes is the same as the number of teacher nodes (m l = n l ), and each student's weight vector w j is close to a corresponding teacher w * j ??? , then the dynamics of Eqn.

2 yields (recovery) convergence w j ??? w * j ??? (Thm. 4).

Furthermore, such convergence is super-linear (i.e., the convergence rate is higher when the weights are closer).??? In the over-parameterization setting (n l > m l ), we show that in the 2-layer case, with the help of toplayer, the portion of weights W u that are close to teacher W * converge (W u ??? W * ).

For other irrelevant weights, while their final values heavily depends on initialization, with the help of top-down modulation, their fan-out top-layer weights converge to zero, and thus have no influence on the network output.

To make Theorem 4 and Theorem 5 work, we make Assumption 3 that the activation field of different teacher nodes should be well-separated.

To justify this, we analyze the BatchNorm bias of pre-trained VGG11 and VGG16.

We check the BatchNorm bias c 1 as both VGG11 and VGG16 use Linear-BatchNorm-ReLU architecture.

After BatchNorm first normalizes the input data into zero mean distribution, the BatchNorm bias determines how much data pass the ReLU threshold.

If the bias is negative, then a small portion of data pass ReLU gating and Assumption 3 is likely to hold.

FIG5 , it is quite clear that the majority of BatchNorm bias parameters are negative, in particular for the top layers.

We evaluate both the fully connected (FC) and ConvNet setting.

For FC, we use a ReLU teacher network of size 50-75-100-125.

For ConvNet, we use a teacher with channel size 64-64-64-64.

The student networks have the same depth but with 10x more nodes/channels at each layer, such that they are substnatially over-parameterized.

When BatchNorm is added, it is added after ReLU.We use random i.i.d Gaussian inputs with mean 0 and std 10 (abbreviated as GAUS) and CIFAR-10 as our dataset in the experiments.

GAUS generates infinite number of samples while CIFAR-10 is a finite dataset.

For GAUS, we use a random teacher network as the label provider (with 100 classes).

To make sure the weights of the teacher are weakly overlapped, we sample each entry of w * j from [???0.5, ???0.25, 0, 0.25, 0.5], making sure they are non-zero and mutually different within the same layer, and sample biases from U [???0.5, 0.5].

In the FC case, the data dimension is 20 while in the ConvNet case it is 16 ?? 16.

For CIFAR-10 we use a pre-trained teacher network with BatchNorm.

In the FC case, it has an accuracy of 54.95%; for ConvNet, the accuracy is 86.4%.

We repeat 5 times for all experiments, with different random seed and report min/max values.

Two metrics are used to check our prediction that some lucky student nodes converge to the teacher:Normalized correlation??.

We compute normalized correlation (or cosine similarity) ?? between teacher and student activations evaluated on a validation set.

At each layer, we average the best correlation over teacher nodes: ?? = mean j ??? max j ?? jj ??? , where ?? jj ??? is computed for each teacher and student pairs (j, j??? )

.?? ??? 1 means that most teacher nodes are covered by at least one student.

Mean Rankr.

After training, each teacher node j??? has the most correlated student node j. We check the correlation rank of j, normalized to [0, 1] (0=rank first), back at initialization and at different epoches, and average them over teacher nodes to yield mean rankr.

Smallr means that student nodes that initially correlate well with the teacher keeps the lead toward the end of training.

Experiments are summarized in Fig. 5 and FIG3 .?? indeed grows during training, in particular for low layers that are closer to the input, where?? moves towards 1.

Furthermore, the final winning student nodes also have a good rank at the early stage of training.

BatchNorm helps a lot, in particular for the CNN case with GAUS dataset.

For CIFAR-10, the final evaluation accuracy (see Appendix) learned by the student is often ??? 1% higher than the teacher.

Using BatchNorm accelerates the growth of accuracy, improvesr, but seems not to accelerate the growth of??.

The theory also predicts that the top-down modulation ?? helps the convergence.

For this, we plot ?? * jj ??? at different layers during optimization on GAUS.

For better visualization, we align each student node index j with a teacher node j??? according to highest ??.

Despite the fact that correlations are computed from the low-layer weights, it matches well with the top-layer modulation (identity matrix structure in FIG0 ).

More ablation studies are in Sec. 8.

We propose a novel mathematical framework for multilayered ReLU networks.

This could tentatively explain many puzzling empirical phenomena in deep learning. .

Correlation?? and mean rankr over training on GAUS.?? steadily grows andr quickly improves over time.

Layer-0 (the lowest layer that is closest to the input) shows best match with teacher nodes and best mean rank.

BatchNorm helps achieve both better correlation and lowerr, in particular for the CNN case.

[5] Simon S Du, Jason D Lee, Yuandong Tian, Barnabas Poczos, and Aarti Singh.

Gradient descent learns onehidden-layer cnn: Don't be afraid of spurious local minima.

ICML, 2018.[6] Jonathan Frankle and Michael Carbin.

The lottery ticket hypothesis: Training pruned neural networks.

ICLR, 2019.[7] Jonathan Frankle, Gintare Karolina Dziugaite, Daniel M Roy, and Michael Carbin.

The lottery ticket hypothesis at scale.

arXiv preprint arXiv:1903.01611, 2019.[8]

Song Han, Jeff Pool, John Tran, and William Dally.

Learning both weights and connections for efficient neural network.

In Advances in neural information processing systems, pages 1135-1143, 2015.[9] Babak Hassibi, David G Stork, and Gregory J Wolff.

Optimal brain surgeon and general network pruning.

In IEEE international conference on neural networks, pages 293-299.

IEEE, 1993.[10] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.

Deep residual learning for image recognition.

In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770-778, 2016.[ 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 378 379 380 381 382 383 384 5.

Appendix: Mathematical Framework Notation.

Consider a student network and its associated teacher network ( FIG0 ).

Denote the input as x. For each node j, denote f j (x) as the activation, f j (x) as the ReLU gating, and g j (x) as the backpropagated gradient, all as functions of x. We use the superscript??? to represent a teacher node (e.g., j ??? ).

Therefore, g j ??? never appears as teacher nodes are not updated.

We use w jk to represent weight between node j and k in the student network.

Similarly, w * j ??? k ??? represents the weight between node j??? and k ??? in the teacher network.

We focus on multi-layered ReLU networks.

We use the following equality extensively: ??(x) = ?? (x)x.

For ReLU node j, we use E j ??? {x : f j (x) > 0} as the activation region of node j.

Objective.

We assume that both the teacher and the student output probabilities over C classes.

We use the output of teacher as the input of the student.

At the top layer, each node c in the student corresponds to each node c ??? in the teacher.

Therefore, the objective is: DISPLAYFORM0 By the backpropagation rule, we know that for each sample x, the (negative) gradient DISPLAYFORM1 The gradient gets backpropagated until the first layer is reached.

Note that here, the gradient g c (x) sent to node c is correlated with the activation of the corresponding teacher node f c ??? (x) and other student nodes at the same layer.

Intuitively, this means that the gradient "pushes" the student node c to align with class c??? of the teacher.

If so, then the student learns the corresponding class well.

A natural question arises:Are student nodes at intermediate layers correlated with teacher nodes at the same layers?One might wonder this is hard since the student's intermediate layer receives no direct supervision from the corresponding teacher layer, but relies only on backpropagated gradient.

Surprisingly, the following theorem shows that it is possible for every intermediate layer: DISPLAYFORM2 .

If all nodes j at layer l satisfies Eqn.

4 DISPLAYFORM3 then all nodes k at layer l ??? 1 also satisfies Eqn.

4 with ?? * kk ??? (x) and ?? kk (x) defined as follows: DISPLAYFORM4 Note that this formulation allows different number of nodes for the teacher and student.

In particular, we consider the over-parameterization setting: the number of nodes on the student side is much larger (e.g., 5-10x) than the number of nodes on the teacher side.

Using Theorem 1, we discover a novel and concise form of gradient update rule: Assumption 1 (Separation of Expectations).

DISPLAYFORM5 DISPLAYFORM6 Theorem 2.

If Assumption 1 holds, the gradient dynamics of deep ReLU networks with objective (Eqn.

3) is: DISPLAYFORM7 Here we explain the notations.

DISPLAYFORM8 We can define similar notations for W (which has n l columns/filters), ??, D, H and L FIG4

In the following, we will use Eqn.

8 to analyze the dynamics of the multi-layer ReLU networks.

For convenience, we first define the two functions ?? l and ?? d (?? is the ReLU function): DISPLAYFORM0 We assume these two functions have the following property .

Assumption 2 (Lipschitz condition).

There exists K d and K l so that: DISPLAYFORM1 Using this, we know that DISPLAYFORM2 , and so on.

For brevity, denote d * * DISPLAYFORM3 is heavy) and so on.

We impose the following assumption: Assumption 3 (Small Overlap between teacher nodes).

There exists l 1 and d 1 so that: DISPLAYFORM4 Intuitively, this means that the probability of the simultaneous activation of two teacher nodes j 1 and j 2 is small.

One such case is that the teacher has negative bias, which means that they cut corners in the input space FIG4 .

We have empirically verified that the majority of biases in BatchNorm layers (after the data are whitened) are negative in VGG11/16 trained on ImageNet (Sec. 3.1).

Batch Normalization (13) has been extensively used to speed up the training, reduce the tuning efforts and improve the test performance of neural networks.

Here we use an interesting property of BatchNorm: the total "energy" of the incoming weights of each node j is conserved over training iterations: Theorem 3 (Conserved Quantity in Batch Normalization).

For Linear ??? ReLU ??? BN or Linear ??? BN ??? ReLU configuration, w j of a filter j before BN remains constant in training.

FIG0 .See Appendix for the proof.

This may partially explain why BN has stabilization effect: energy will not leak from one layer to nearby ones.

Due to this property, in the following, for convenience we assume w j 2 = w * j 2 = 1, and the gradient??? j is always orthogonal to the current weight w j .

Note that on the teacher side we can always push the magnitude component to the upper layer; on the student side, random initialization naturally leads to constant magnitude of weights.

If n l = m l , L * l = L l = I (e.g., the input of layer l is whitened) and ?? * l+1 = ?? l+1 = 11 T (all ?? entries are 1), then the following theorem shows that weight recovery could follow (we use j as j ??? ).

442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 486 487 488 489 490 491 492 493 494 Teacher Student Convergence Figure 8 .

Over-parameterization and top-down modulation.

Thm.

5 shows that under certain conditions, the relevant weights Wu ??? W * and weights connecting to irrelevant student nodes Vr ??? 0.

DISPLAYFORM0 See Appendix for the proof.

Here we list a few remarks:Faster convergence near w * j .

we can see that due to the fact that h * jj in general becomes larger when w j ??? w * j (since cos ?? 0 can be close to 1), we expect a super-linear convergence near w * j .

This brings about an interesting winner-take-all mechanism: if the initial overlap between a student node j and a particular teacher node is large, then the student node will snap to it ( FIG0 ).Importance of projection operator P ??? wj .

Intuitively, the projection is needed to remove any ambiguity related to weight scaling, in which the output remains constant if the top-layer weights are multiplied by a constant ??, while the low-layer weights are divided by ??.

Previous works (5) also uses similar techniques while we justify it with BN.

Without P ??? wj , convergence can be harder.

In the over-parameterization case (n l > m l , e.g., 5-10x), we arrange the variables into two parts: W = [W u , W r ], where W u contains m l columns (same size as W * ), while W r contains n l ??? m l columns.

We use [u] (or u-set) to specify nodes 1 ??? j ??? m, and [r] (or r-set) for the remaining part.

In this case, if we want to show "the main component" W u converges to W * , we will meet with one core question: to where will W r converge, or whether W r will even converge at all?

We need to consider not only the dynamics of the current layer, but also the dynamics of the upper layer.

See Appendix for the proof (and definition of?? in Eqn.

47).

The intuition is: if W u is close to W * and W r are far away from them due to Assumption 3, the off-diagonal elements of L and L * are smaller than diagonal ones.

This causes V u to move towards V * and V r to move towards zero.

When V r becomes small, so does ?? jj for j ??? [r] or j ??? [r].

This in turn suppresses the effect of W r and accelerates the convergence of W u .

V r ??? 0 exponentially so that W r stays close to its initial locations, and Assumption 3 holds for all iterations.

A few remarks:Flat minima.

Since V r ??? 0, W r can be changed arbitrarily without affecting the outputs of the neural network.

This could explain why there are many flat directions in trained networks, and why many eigenvalues of the Hessian are close to zero (28).Understanding of pruning methods.

Theorem 5 naturally relates two different unstructured network pruning approaches: pruning small weights in magnitude (8; 6) and pruning weights suggested by Hessian (18; 9).

It also suggests a principled structured pruning method: instead of pruning a filter by checking its weight norm, pruning accordingly to its top-down modulation.

Accelerated convergence and learning rate schedule.

For simplicity, the theorem uses a uniform (and conservative) 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 540 541 542 543 544 545 546 547 548 ?? throughout the iterations.

In practice, ?? is initially small (due to noise introduced by r-set) but will be large after a few iterations when V r vanishes.

Given the same learning rate, this leads to accelerated convergence.

At some point, the learning rate ?? becomes too large, leading to fluctuation.

In this case, ?? needs to be reduced.

Many-to-one mapping.

Theorem 5 shows that under strict conditions, there is one-to-one correspondence between teacher and student nodes.

In general this is not the case.

Two students nodes can be both in the vicinity of a teacher node w * j and converge towards it, until that node is fully explained.

We leave it to the future work for rigid mathematical analysis of many-to-one mappings.

Random initialization.

One nice thing about Theorem 5 is that it only requires the initial W u ??? W * to be small.

In contrast, there is no requirement for small V r .

Therefore, we could expect that with more over-parameterization and random initialization, in each layer l, it is more likely to find the u-set (of fixed size m l ), or the lucky weights, so that W u is quite close to W * .

At the same time, we don't need to worry about W r which grows with more over-parameterization.

Moreover, random initialization often gives orthogonal weight vectors, which naturally leads to Assumption 3.

Using a similar approach, we could extend this analysis to multi-layer cases.

We conjecture that similar behaviors happen: for each layer, due to over-parameterization, the weights of some lucky student nodes are close to the teacher ones.

While these converge to the teacher, the final values of others irrelevant weights are initialization-dependent.

If the irrelevant nodes connect to lucky nodes at the upper-layer, then similar to Thm.

5, the corresponding fan-out weights converge to zero.

On the other hand, if they connect to nodes that are also irrelevant, then these fan-out weights are not-determined and their final values depends on initialization.

However, it doesn't matter since these upper-layer irrelevant nodes eventually meet with zero weights if going up recursively, since the top-most output layer has no over-parameterization.

We leave a formal analysis to future work.

Proof.

The first part of gradient backpropagated to node j is: ) 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 594 595 596 597 598 599 600 601 602 603 Therefore, for the gradient to node k, we have: DISPLAYFORM0 DISPLAYFORM1 DISPLAYFORM2 And similar for ?? kk (x).

Therefore, by mathematical induction, we know that all gradient at nodes in different layer follows the same form.

Proof.

Using Thm.

1, we can write down weight update for weight w jk that connects node j and node k: DISPLAYFORM0 Note that j ??? , k ??? , j and k run over all parents and children nodes on the teacher side.

This formulation works for overparameterization (e.g., j??? and j can run over different nodes).

Applying Assumption 1 and rearrange terms in matrix form yields Eqn.

8.

Proof.

Given a batch with size N , denote pre-batchnorm activations as DISPLAYFORM0 T and gradients as DISPLAYFORM1 T (See FIG0 ).f = (f ??? ??)/?? is its whitened version, and c 0f + c 1 is the final output of BN.

Here ?? = DISPLAYFORM2 2 and c 1 , c 0 are learnable parameters.

With vector notation, the gradient update in BN has a compact form with clear geometric meaning:Lemma 1 (Backpropagation of Batch Norm (34)).

For a top-down gradient g, BN layer gives the following gradient update (P ??? f ,1 is the orthogonal complementary projection of subspace {f , 1}): DISPLAYFORM3 Intuitively, the back-propagated gradient J BN (f )g is zero-mean and perpendicular to the input activation f of BN layer, as illustrated in FIG0 .

Unlike (15; 37) that analyzes BN in an approximate manner, in Thm.

1 we do not impose any assumptions.

Given Lemma 1, we can prove Thm.

3. FIG0 , using the property that E x g lin j f lin j = 0 (the expectation is taken over batch) and the weight update rule??? jk = E x g lin j f k (over the same batch), we have: DISPLAYFORM4 For FIG0 , note that DISPLAYFORM5 rl j = 0 and conclusion follows.

For simplicity, in the following, we use ??w j = w j ??? w * j .

Lemma 2 (Bottom Bounds).

Assume all w j = w j = 1.

Denote DISPLAYFORM0 If Assumption 2 holds, we have: DISPLAYFORM1 If Assumption 3 also holds, then: DISPLAYFORM2 Proof.

We have for j = j : DISPLAYFORM3 If Assumption 3 also holds, we have: DISPLAYFORM4 Lemma 3 (Top Bounds).

Denote DISPLAYFORM5 If Assumption 2 holds, we have: DISPLAYFORM6 If Assumption 3 also holds, then: DISPLAYFORM7 Proof.

The proof is similar to Lemma 2.Lemma 4 (Quadratic fall-off for diagonal elements of L).

For node j, we have: DISPLAYFORM8 Proof.

The intuition here is that both the volume of the affected area and the weight difference are proportional to ??w j .

l * jj ??? l jj is their product and thus proportional to ??w j 2 .

See FIG0 .

Proof.

First of all, note that ??w j = 2 sin ??j 2 ??? 2 sin ??0 2 .

So given ?? 0 , we also have a bound for ??w j .

When ?? = ?? * = 11 T , the matrix form can be written as the following: DISPLAYFORM0 by using P ??? wj w j ??? 0 (and thus h jj doesn't matter).

Since w j is conserved, it suffices to check whether the projected weight vector P ??? w * j w j of w j onto the complementary space of the ground truth node w * j , goes to zero: DISPLAYFORM1 Denote ?? j = ???(w j , w * j ) and a simple calculation gives that sin ?? j = P ??? w * j w j .

First we have: DISPLAYFORM2 From Lemma 2, we knows that DISPLAYFORM3 Note that here we have ??w j = 2 sin DISPLAYFORM4 We discuss finite step with very small learning rate ?? > 0: DISPLAYFORM5 Here DISPLAYFORM6 is an iteration independent constant.

We set ?? = cos DISPLAYFORM7 jj and from Lemma 2 we know d * jj ???d for all j.

Then given the inductive hypothesis that sin ?? t j ??? (1 ??? ??d??) t???1 sin ?? 0 , we have: DISPLAYFORM8 Therefore, sin ?? DISPLAYFORM9 The proof can be decomposed in the following three lemma.

Lemma 5 (Top-layer contraction).

If (W-Separation) holds for t, then (V -Contraction)) holds for iteration t + 1.Lemma 6 (Bottom-layer contraction).

If (V -Contraction) holds for t, then (W u -Contraction) holds for t + 1 and (W r -Bound) holds for t + 1.Lemma 7 (Weight separation).

If (W-Separation) holds for t, (W r -Bound) holds for t + 1 and (W u -Contraction) holds for t + 1, then (W-Separation) holds for t + 1.As suggested by FIG0 , if all the three lemmas are true then the induction hypothesis are true.

In the following, we will prove the three lemmas.7.6.1.

LEMMA 5Proof.

On the top-layer, we haveV = L DISPLAYFORM10 , where v j is the j-th row of the matrix V .For each component, we can write: DISPLAYFORM11 Note that there is no projection (if there is any, the projection should be in the columns rather than the rows).If (W-Separation) is true, we know that for j = j , DISPLAYFORM12 Now we discuss j ??? [u] and j ??? [r]:Relevant nodes.

For j ??? [u], the first two terms are: DISPLAYFORM13 From Lemma 4 we know that: ) 882 883 884 885 886 887 888 889 890 891 892 893 894 895 896 897 898 899 900 901 902 903 904 905 906 907 908 909 910 911 912 913 914 915 916 917 918 919 920 921 922 923 924 925 926 927 928 929 930 931 932 933 initial value is DISPLAYFORM14 DISPLAYFORM15 Therefore, we prove that (W r -Bound) holds for iteration t + 1.

Proof.

Simply followed from combining Lemma 3, Lemma 2 and weight bounds (W u -Contraction) and (V -Contraction).

Besides, we also perform ablation studies on GAUS.Size of teacher network.

As shown in FIG0 , for small teacher networks (FC 10-15-20-25), the convergence is much faster and training without BatchNorm is faster than training with BatchNorm.

For large teacher networks, BatchNorm definitely increases convergence speed and growth of??.

Finite versus Infinite Dataset.

We also repeat the experiments with a pre-generated finite dataset of GAUS in the CNN case, and find that the convergence of node similarity stalls after a few iterations.

This is because some nodes receive very few data points in their activated regions, which is not a problem for infinite dataset.

We suspect that this is probably the reason why CIFAR-10, as a finite dataset, does not show similar behavior as GAUS .

937 938 939 940 941 942 943 944 945 946 947 948 949 950 951 952 953 954 955 956 957 958 959 960 961 962 963 964 965 966 967 968 969 970 971 972 973 974 975 976 977 978 979 980 981 982 983 984 985 986 987 988 Layer-0Layer-1Layer-2Layer-3 ??* at initialization ??* after optimization H* after optimization FIG0 .

Visualization of (transpose of) H * and ?? * matrix before and after optimization (using GAUS).

Student node indices are reordered according to teacher-student node correlations.

After optimization, student node who has high correlation with the teacher node also has high ?? entries.

Such a behavior is more prominent in H * matrix that combines ?? * and the activation patterns D * of student nodes (Sec. 5).

@highlight

A theoretical framework for deep ReLU network that can explains multiple puzzling phenomena like over-parameterization, implicit regularization, lottery tickets, etc. 