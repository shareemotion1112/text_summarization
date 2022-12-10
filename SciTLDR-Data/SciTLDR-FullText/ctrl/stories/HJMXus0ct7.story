We propose a new approach, known as the iterative regularized dual averaging (iRDA), to improve the efficiency of convolutional neural networks (CNN) by significantly reducing the redundancy of the model without reducing its accuracy.

The method has been tested for various data sets, and proven to be significantly more efficient than most existing compressing techniques in the deep learning literature.

For many popular data sets such as MNIST and CIFAR-10, more than 95% of the weights can be zeroed out without losing accuracy.

In particular, we are able to make ResNet18 with 95% sparsity to have an accuracy that is comparable to that of a much larger model ResNet50 with the best 60% sparsity as reported in the literature.

In recent decades, deep neural network models have achieved unprecedented success and state-ofthe-art performance in various tasks of machine learning or artificial intelligence, such as computer vision, natural language processing and reinforcement learning BID11 .

Deep learning models usually involve a huge number of parameters to fit variant kinds of datasets, and the number of data may be much less than the amount of parameters BID9 .

This may implicate that deep learning models have too much redundancy.

This can be validated by the literatures from the general pruning methods BID18 to the compressing models BID6 .While compressed sensing techniques have been successfully applied in many other problems, few reports could be found in the literature for their application in deep learning.

The idea of sparsifying machine learning models has attracted much attention in the last ten years in machine learning BID2 ; BID22 .

When considering the memory and computing cost for some certain applications such as Apps in mobile, the sparsity of parameters plays a very important role in model compression BID6 ; BID0 .

The topic of computing sparse neural networks can be included in the bigger topic on the compression of neural networks, which usually further involves the speedup of computing the compressed models.

There are many sparse methods in machine learning models such as FOBOS method BID3 , also known as proximal stochastic gradient descent (prox-SGD) methods BID16 , proposed for general regularized convex optimization problem, where 1 is a common regularization term.

One drawback of prox-SGD is that the thresholding parameters will decay in the training process, which results in unsatisfactory sparsity BID22 .

Apart from that, the regularized dual averaging (RDA) method BID22 , proposed to obtain better sparsity, has been proven to be convergent with specific parameters in convex optimization problem, but has not been applied in deep learning fields.

In this paper, we analyze the relation between simple dual averaging (SDA) method BID17 and the stochastic gradient descent (SGD) method BID19 , as well as the relation between SDA and RDA.

It is well-known that SGD and its variants work quite well in deep learning problems.

However, there are few literatures in applying pure training algorithms to deep CNNs for model sparsification.

We propose an iterative RDA (iRDA) method for training sparse CNN models, and prove the convergence under convex conditions.

Numerically, we compare prox-SGD with iRDA, where the latter can achieve better sparsity results while keeping satisfactory accuracy on MNIST, CIFAR-10 and CIFAR-100.

We also show iRDA works for different CNN models such as VGG BID21 and BID9 .

Finally, we compare the performance of iRDA with some other state-of-the-art compression methods.

BID0 reviews the work on compressing neural network models, and categorizes the related methods into four schemes: parameter pruning and sharing, low-rank factorization, transfered/compact convolutional filters and knowledge distillation.

Among them, BID14 uses sparse decomposition on the convolutional filters to get sparse neural networks, which could be classified to the second scheme.

Apart from that, BID7 prunes redundant connections by learning only the important parts.

BID15 starts from a Bayesian point of view, and removes large parts of the network through sparsity inducing priors.

BID23 BID10 combines reinforcement learning methods to compression.

BID13 considers deep learning as a discrete-time optimal control problem, and obtains sparse weights on ternary networks.

Recently, BID4 applies RDA to fully-connected neural network models on MNIST.

Let z = (x, y) be an input-output pair of data, such as a picture and its corresponding label in a classification problem, and f (w, z) be the loss function of neural networks, i.e. a scalar function that is differentiable w.r.t.

weights w.

We are interested in the expected risk minimization problem DISPLAYFORM0 The empirical risk minimization DISPLAYFORM1 is an approximation of (1) based on some finite given samples {z 1 , z 2 , . . .

, z T } , where T is the size of the sample set.

Regularization is a useful technique in deep learning.

In general, the regularized expected risk minimization has the form DISPLAYFORM2 where Ψ(w) is a regularization term with certain effect.

For example, Ψ(w) = w 2 2 may improve the generalization ability, and an 1 -norm of w can give sparse solutions.

The corresponding regularized empirical risk minimization we concern takes the form DISPLAYFORM3 SDA method is a special case of primal-dual subgradient method first proposed in BID17 .

BID22 proposes RDA for online convex and stochastic optimization.

RDA not only keeps the same convergence rate as Prox-SGD, but also achieves more sparsity in practice.

In next sections, we will discuss the connections between SDA and SGD, as well as RDA and Prox-SGD.

We then propose iRDA for 1 regularized problem of deep neural networks.

As a solution of (2), SDA takes the form DISPLAYFORM0 The first term t τ =1 g τ , w is a linear function obtained by averaging all previous stochastic gradient.

g t is the subgradient of f t .

The second term h(w) is a strongly convex function, and {β t } is a nonnegative and nondecreasing sequence which determines the convergence rate.

As g τ (w τ ), τ = 1, . . . , t − 1 is constant in current iteration, we use g τ instead for simplicity in the following.

Since subproblem equation 5 is strongly convex, it has a unique optimal solution w t+1 .Let w 0 be the initial point, and h(w) = 1 2 w − w 0 2 2 , the iteration scheme of SDA can be written as DISPLAYFORM1 DISPLAYFORM2 .

Let β t = γt α , SDA can be rewritten recursively as DISPLAYFORM3 where 1 − 1 −

For the regularized problem (4), we recall the well-known Prox-SGD and RDA method first.

At each iteration, Prox-SGD solves the subproblem DISPLAYFORM0 Specifically, α t = 1 γ √ t obtains the best convergence rate.

The first two terms are an approximation of the original objective function.

Note that without the regularization term Ψ, equation 8 is equivalent to SGD.

It can be written in forward-backward splitting (FOBOS) scheme DISPLAYFORM1 DISPLAYFORM2 where the forward step is equivalent to SGD, and the backward step is a soft-thresholding operator working on w t+ DISPLAYFORM3 with the soft-thresholding parameter α t .Different from Prox-SGD, each iteration of RDA takes the form DISPLAYFORM4 Similarly, taking h(w) = 1 2 w − w 0 2 2 , RDA can be written as DISPLAYFORM5 = arg min DISPLAYFORM6 or equivalently, DISPLAYFORM7 w t+1 = arg min DISPLAYFORM8 where β t = γ √ t to obtain the best convergence rate.

From equation 14, one can see that the forward step is actually SDA and the backward step is the soft-thresholding operator, with the parameter t/β t .3.3 1 REGULARIZATION AND THE SPARSITY Set Ψ(w) = λ w 1 .

The problem (4) then becomes DISPLAYFORM9 where λ is a hyper-parameter that determines sparsity.

In this case, from Xiao's analysis of RDA Xiao (2010) , the expected cost Eφ(w t ) − φ associated with the random variablew t converges with rate O( DISPLAYFORM10 This convergence rate is consistent with FOBOS Duchi and Singer (2009) .

However, both results assume f to be a convex function, which can not be guaranteed in deep learning.

Nevertheless, we can still verify that RDA is a powerful sparse optimization method for deep neural networks.

We conclude the closed form solutions of Prox-SGD and RDA for equation 16 as follows.

has the closed form solution DISPLAYFORM0 2.

The subproblem of RDA DISPLAYFORM1 has the closed form solution DISPLAYFORM2 3.

The √ t-proximal stochastic gradient method has the form DISPLAYFORM3 The difference between √ t-Prox-SGD and Prox-SGD is the soft-thresholding parameter chosen to be √ t. It has the closed form solution DISPLAYFORM4 It is equivalent to DISPLAYFORM5 where the objective function is actually an approximation of DISPLAYFORM6 We can easily conclude that this iteration will converge to w = 0 if DISPLAYFORM7 Now compare the threshold λ P G = α t λ of PG and the threshold λ RDA = t βt λ of RDA.

With DISPLAYFORM8 and β t = γ √ t, we have λ P G → 0 and λ RDA → ∞ as t → 0.

It is clear that RDA uses a much more aggressive threshold, which guarantees to generate significantly more sparse solutions.

Note that when Ψ = λ w 1 , RDA requires w 1 = w 0 = 0.

However, this will make deep neural network a constant function, with which the parameters can be very hard to update.

Thus, in Algorithm 1, we modify the RDA method as Step 1, where w 1 can be chosen not equal to 0, and add an extra Step 2 to improve the performance.

We also prove the convergence rate of Step 1 for convex problem is O( DISPLAYFORM0 Theorem 3.1 Assume there exists an optimal solution w to the problem (3) with Ψ(w) = λ w 1 that satisfies h(w ) ≤ D 2 for some D > 0, and let φ = φ(w ).

Let the sequences {w t } t≥1 be generated by Step 1 in iRDA, and assume g t * ≤ G for some constant G. Then the expected cost Eφ(w t ) converges to φ with rate O( DISPLAYFORM1 See Appendix A for the proof.

To apply iRDA, the weights of a neural network should be initialized differently from that in a normal optimization method such as SGD or its variants.

Our initialization is based on BID12 , BID5 and BID8 , with an additional re-scaling.

Let s be a scalar, the mean and the standard deviation of the uniform distribution for iRDA is zero and DISPLAYFORM0 respectively, where c is the number of channels, and k is the spatial filter size of the layer (see BID8 ).Choosing a suitable s is important when applying iRDA.

As shown in TAB2 and TAB3 in Appendix B, if s is too small or too large, the training process could be slowed down and the generalization ability may be affected.

Moreover, a small s usually requires much better initial weights, which results in too many samplings in initialization process.

In our experiments, a good s for iRDA is usually much larger than √ 2, and unsuitable for SGD algorithms.

Iterative retraining is a method that only updates the non-zero parameters at each iteration.

A trained model can be further updated with retraining, thus both the accuracies and sparsity can be improved.

See Table 4 for comparisons on CIFAR-10.

The iterative RDA method for 1 regularized DNNs Input:• A strongly convex function h(w) = w 2 2 .•

A nonnegative and nondescreasing sequence β t = γ √ t.

Step 1: RDA with proper initialization Initialize: set w 0 = 0,ḡ 0 = 0 and randomly choose w 1 with methods explained in section 3.5. for t=1,2, ..., T do Given the sample z it and corresponding loss function f it .Compute the stochastic gradient g t = ∇f it (w t ).Update the average gradient:ḡ DISPLAYFORM0 Compute the next weight vector: DISPLAYFORM1 Step 2: iterative retraining for t=T+1,T+2,T+3, ... do Given the sample z it and corresponding loss function f it .Compute the stochastic gradient DISPLAYFORM2 Set (g t ) j = 0 if (w t ) j = 0 for every j. Update the average gradient:ḡ DISPLAYFORM3 Compute the next weight vector: DISPLAYFORM4

In this section, σ denotes the sparsity of a model, i.e. σ = quantity of zero parameters quantity of all parameters .All neural networks are trained with mini-batch size 128.

We provide a test on different hyper-parameters, so as to give an overview of their effects on performance, as shown in TAB4 .

We also show that the sparsity and the accuracy can be balanced with iRDA by adjusting the parameters λ and γ, as shown in TAB6 .

Both tables are put in Appendix C.

We compare iRDA with several methods including prox-SGD, √ t−SGD and normal SGD, on different datasets including MNIST, CIFAR-10, CIFAR-100 and ImageNet(ILSVRC2012).

The main results are shown in TAB0 .

Table 2 shows the performance of iRDA on different architectures including ResNet18, VGG16 and VGG19.

TAB1 shows the performance of iRDA on different Figure 1 : The first 120 epochs of loss curves corresponding to TAB0 , and the sparsity curve for another result, where the top-1 validation accuracy is 91.34%, and σ = 0.87.

datasets including MNIST, CIFAR-10, CIFAR-100 and ImageNet(ILSVRC2012).

In all tables, SGD denotes stochastic gradient methods with momentum Ruder (2016).

Currently, many compression methods include human experts involvement.

Some methods try to combine other structures in training process to automatize the compression process.

For example, BID10 combines reinforcement learning.

iRDA, as an algorithm, requires no extra structure.

As shown above, iRDA can achieve good sparsity while keeping accuracy automatically, with carefully chosen parameters.

For CIFAR-10, we compare the performance of iRDA with some other state-of-art compression methods in Table 4 .

Due to different standards, σ is referred to directly or computed from the original papers approximately.

In comparison with many existing rule-based heuristic approaches, the new approach is based on a careful and iterative combination of 1 regularization and some specialized training algorithms.

We find that the commonly used training algorithms such as SGD methods are not effective.

We thus develop iRDA method that can be used to achieve much better sparsity.

iRDA is a variant of RDA methods that have been used for some special types of online convex optimization problems in the literature.

New elements in the iRDA mainly consist of judicious initialization and iterative retraining.

In addition, iRDA method is carefully analyzed on its convergence for convex objective functions.

Many deep neural networks trained by iRDA can achieve good sparsity while keeping the same validation accuracy as those trained by SGD with momentum on many popular datasets.

This result shows iRDA is a powerful sparse optimization method for image classification problems in deep learning fields.

One of the differences between RDA Xiao (2010) and iRDA is that the former one takes w 1 = arg min w h(w) whereas the latter one chooses w 1 randomly.

In the following, we will prove the convergence of iRDA Step 1 for convex problem.

The proofs use Lemma 9, Lemma 10, Lemma 11 directly and modify Theorem 1 and Theorem 2 in BID22 .

For clarity, we have some general assumptions:• The regularization term Ψ(w) is a closed convex function with convexity parameter σ and domΨ is closed.• For each t ≥ 1, f t (w) is convex and subdifferentiable on domΨ.• h(w) is strongly convex on domΨ and subdifferentiable on rint(domΨ) and also satisfies DISPLAYFORM0 Without loss of generality, assume h(w) has convexity parameter 1 and min w h(w) = 0.• There exist a constant G such that DISPLAYFORM1 • Require {β} t≥1 be a nonnegative and nondecreasing sequence and DISPLAYFORM2 Moreover, we could always choose β 1 ≥ σ such that β 0 = β 1 .•

For a random choosing w 1 , we assume DISPLAYFORM3 First of all, we define two functions: DISPLAYFORM4 DISPLAYFORM5 The maximum in (37) is always achieved because F D = {w ∈ domΨ|h(w) ≤ D 2 } is a nonempty compact set.

Because of (35), we have σt+β t ≥ β 0 > 0 for all t ≥ 0, which means tΨ(w)+β t h(w) are all strongly convex, therefore the maximum in (38) is always achieved and unique.

As a result, we have domU t = domV t = E * for all t ≥ 0.

Moreover, by the assumption (33), both of the functions are nonnegative.

Let s t denote the sum of the subgradients obtained up to time t in iRDA Step 1, that is DISPLAYFORM6 and π t (s) denotes the unique maximizer in the definition of V t (s) DISPLAYFORM7 which then gives DISPLAYFORM8 Lemma A.1 For any s ∈ E * and t ≥ 0, we have DISPLAYFORM9 For a proof, see Lemma 9 in Xiao (2010).Lemma A.2 The function V t is convex and differentiable.

Its gradient is given by DISPLAYFORM10 and the gradient Lipschitz continuous with constant 1/(σt + β t ), that is DISPLAYFORM11 Moreover, the following inequality holds: DISPLAYFORM12 The results are from Lemma 10 in BID22 .Lemma A.3 For each t ≥ 1, we have DISPLAYFORM13 Since h(w t+1 ) ≥ 0 and the sequence {β t } t≥1 is nondecreasing, we have DISPLAYFORM14 DISPLAYFORM15 To prove this lemma, we refer to the Lemma 11 in Xiao (2010).

What's more, from the assumption 35, we could always choose β 1 ≥ σ such that β 1 = β 0 and DISPLAYFORM16 The learner's regret of online learning is the difference between his cumulative loss and the cumulative loss of the optimal fixed hypothesis, which is defined by DISPLAYFORM17 and bounded by DISPLAYFORM18 Lemma A.4 Let the sequence {w t } t≥1 and {g t }

t≥1 be generated by iRDA Step 1, and assume FORMULA2 and FORMULA2 hold.

Then for any t ≥ 1 and any DISPLAYFORM19 Proof First, we define the following gap sequence which measures the quality of the solutions w 1 , .., w t : DISPLAYFORM20 and δ t is an upper bound on the regret R t (w) for all w ∈ F D , to see this, we use the convexity of f t (w) in the following: DISPLAYFORM21 Then, We are going to derive an upper bound on δ t .

For this purpose, we subtract t τ =1 g τ , w 0 in (53), which leads to DISPLAYFORM22 the maximization term in (55) is in fact U t (−s t ), therefore, by applying Lemma A.1, we have DISPLAYFORM23 Next, we show that ∆ t is an upper bound for the right-hand side of inequality (56).

We consider τ ≥ 2 and τ = 1 respectively.

For any τ ≥ 2, we have DISPLAYFORM24 where FORMULA3 , FORMULA2 , FORMULA3 and FORMULA2 are used.

Therefore, we have DISPLAYFORM25 , ∀τ ≥ 2.For τ = 1, we have a similar inequality by using (49) DISPLAYFORM26 Summing the above inequalities for τ = 1, ..., t and noting that V 0 (−s 0 ) = V 0 = 0, we arrive at DISPLAYFORM27 Since Ψ(w t+1 ) ≥ 0, we subtract it from the left hand side and add Ψ(w 1 ) to both sides of the above inequality yields DISPLAYFORM28 Combing FORMULA3 , FORMULA4 , (57) and using assumption (34) and (36)we conclude DISPLAYFORM29 Lemma A.5 Assume there exists an optimal solution w to the problem (3) that satisfies h(w ) ≤ D 2 for some D > 0, and let φ = φ(w ).

Let the sequences {w t } t≥1 be generated by iRDA Step 1, and assume g t * ≤ G for some constant G. Then for any t ≥ 1, the expected cost associated with the random variablew t is bounded as DISPLAYFORM30 Proof First, from the definition (50), we have the regret at w DISPLAYFORM31 Let z[t] denote the collection of i.i.d.

random variables (z , ..., z t ).

We note that the random variable w τ , where 1 ≤ w ≥ t, is a function of (z 1 , ..., z τ −1 ) and is independent of (z τ , ..., z t ).

Therefore DISPLAYFORM32 and DISPLAYFORM33 Since φ = φ(w ) = min w φ(w), we have the expected regret DISPLAYFORM34 Then, by convexity of φ, we have DISPLAYFORM35 Finally, from FORMULA4 and FORMULA4 , we have DISPLAYFORM36 Then the desired follows from that of Lemma A.4.

Proof of Theorem 3.1 From Lemma A.5, the expected cost associated with the random variablew t is bounded as DISPLAYFORM37 Here, we consider 1 regularization function Ψ(w) = λ w 1 and it is a convex but not strongly convex function, which means σ = 0.

Now, we consider how to choose β t for t ≥ 1 and β 0 = β 1 .

First if β t = γt, we have 1 t · γtD 2 = γD 2 , which means the expected cost does not converge.

Then assume β t = γt α , α > 0 and α = 1, the right hand side of the inequality (60) becomes DISPLAYFORM38 From above, we see that if 0 < α < 1, the expected cost converges and the optimal convergence rate O(t We have shown why prox-SGD will give poor sparsity, and although √ t-prox-SGD may introduce greater sparsity, it is not convergent.

Finally, iRDA gives the best result, on both the top-1 accuracy and the sparsity.

iRDA (

<|TLDR|>

@highlight

A sparse optimization algorithm for deep CNN models.