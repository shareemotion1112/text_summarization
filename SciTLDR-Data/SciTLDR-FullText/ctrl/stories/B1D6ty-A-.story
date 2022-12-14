We present DANTE, a novel method for training neural networks, in particular autoencoders, using the alternating minimization principle.

DANTE provides a distinct perspective in lieu of traditional gradient-based backpropagation techniques commonly used to train deep networks.

It utilizes an adaptation of quasi-convex optimization techniques to cast autoencoder training as a bi-quasi-convex optimization problem.

We show that for autoencoder configurations with both differentiable (e.g. sigmoid) and non-differentiable (e.g. ReLU) activation functions, we can perform the alternations very effectively.

DANTE effortlessly extends to networks with multiple hidden layers and varying network configurations.

In experiments on standard datasets, autoencoders trained using the proposed method were found to be very promising when compared to those trained using traditional backpropagation techniques, both in terms of training speed, as well as feature extraction and reconstruction performance.

For much of the recent march of deep learning, gradient-based backpropagation methods, e.g. Stochastic Gradient Descent (SGD) and its variants, have been the mainstay of practitioners.

The use of these methods, especially on vast amounts of data, has led to unprecedented progress in several areas of artificial intelligence.

On one hand, the intense focus on these techniques has led to an intimate understanding of hardware requirements and code optimizations needed to execute these routines on large datasets in a scalable manner.

Today, myriad off-the-shelf and highly optimized packages exist that can churn reasonably large datasets on GPU architectures with relatively mild human involvement and little bootstrap effort.

However, this surge of success of backpropagation-based methods in recent years has somewhat overshadowed the need to continue to look for options beyond backprogagation to train deep networks.

Despite several advancements in deep learning with respect to novel architectures such as encoderdecoder networks and generative adversarial models, the reliance on backpropagation methods remains.

While reinforcement learning methods are becoming increasingly popular, their scope is limited to a particular family of settings such as agent-based systems or reward-based learning.

Recent efforts have studied the limitations of SGD-based backpropagation, including parallelization of SGDbased techniques that are inherently serial BID14 ); vanishing gradients, especially for certain activation functions BID7 ); convergence of stochastic techniques to local optima BID0 ); and many more.

For a well-referenced recent critique of gradient-based methods, we point the reader to BID14 .From another perspective, there has been marked progress in recent years in the area of non-convex optimization (beyond deep learning), which has resulted in scalable methods such as iterated hard thresholding BID2 ) and alternating minimization BID9 ) as methods of choice for solving large-scale sparse recovery, matrix completion, and tensor factorization tasks.

Several of these methods not only scale well to large problems, but also offer provably accurate solutions.

In this work, we investigate a non-backpropagation strategy to train neural networks, leveraging recent advances in quasi-convex optimization.

Our method is called DANTE (Deep AlterNations for Training autoEncoders), and it offers an alternating minimization-based technique for training neural networks -in particular, autoencoders.

DANTE is based on a simple but useful observation that the problem of training a single hidden-layer autoencoder can be cast as a bi-quasiconvex optimization problem (described in Section 3.1).

This observation allows us to use an alternating optimization strategy to train the autoencoder, where each step involves relatively simple quasi-convex problems.

DANTE then uses efficient solvers for quasiconvex problems including normalized gradient descent BID11 ) and stochastic normalized gradient descent BID6 ) to train autoencoder networks.

The key contributions of this work are summarized below:??? We show that viewing each layer of a neural network as applying an ensemble of generalized linear transformations, allows the problem of training the network to be cast as a bi-quasiconvex optimization problem (exact statement later).??? We exploit this intuition by employing an alternating minimization strategy, DANTE, that reduces the problem of training the layers to quasi-convex optimization problems.??? We utilize the state-of-the-art Stochastic Normalized Gradient Descent (SNGD) technique BID6 ) for quasi-convex optimization to provide an efficient implementation of DANTE for networks with sigmoidal activation functions.

However, a limitation of SNGD is its inability to handle non-differentiable link functions such as the ReLU.??? To overcome this limitation, we introduce the generalized ReLU, a variant of the popular ReLU activation function and show how SNGD may be applied with the generalized ReLU function.

This presents an augmentation in the state-of-the-art in quasi-convex optimization and may be of independent interest.

This allows DANTE to train AEs with both differentiable and non-differentiable activation functions, including ReLUs and sigmoid.??? We show that SNGD offers provably more rapid convergence with the generalized ReLU function than it does even for the sigmoidal activation.

This is corroborated in experiments as well.

A key advantage of our approach is that these theoretical results can be used to set learning rates and batch sizes without finetuning/cross-validation.??? We also show DANTE can be easily extended to train deep AEs with multiple hidden layers.??? We empirically validate DANTE with both the generalized ReLU and sigmoid activations and establish that DANTE provides competitive test errors, reconstructions and classification performance (with the learned representations), when compared to an identical network trained using standard mini-batch SGD-based backpropagation.

Backpropagation-based techniques date back to the early days of neural network research BID13 ; BID4 ) but remain to this day, the most commonly used methods for training a variety of neural networks including multi-layer perceptrons, convolutional neural networks, autoencoders, recurrent networks and the like.

Recent years have seen the development of other methods, predominantly based on least-squares approaches, used to train neural networks.

Carreira-Perpinan and Wang BID3 ) proposed a least-squares based method to train a neural network.

In particular, they introduced the Method of Auxiliary Constraints (MAC), and used quadratic penalties to enforce equality constraints.

BID12 ) proposed an Expectation-Maximization (EM) approach derived from a hierarchical generative model called the Deep Rendering Model (DRM), and also used least-squared parameter updates in each of the EM steps.

They showed that forward propagation in a convolutional neural network was equivalent to the inference on their DRM.

Unfortunately, neither of these methods has publicly available implementations or published training results to compare against.

More recently, Taylor et al. proposed a method to train neural networks using the Alternating Direction Method of Multipliers (ADMM) and Bregman iterations BID14 ).

The focus of this method, however, was on scaling the training of neural networks to a distributed setting on multiple cores across a computing cluster.

Jaderberg also proposed the idea of 'synthetic gradients' in BID8 .

While this approach is interesting, this work is more focused towards a more efficient way to carry out gradient-based parameter updates in a neural network.

In our work, we focus on an entirely new approach to training neural networks -in particular, autoencoders -using alternating optimization, quasi-convexity and SNGD, and show that this approach shows promising results on the a range of datasets.

Although alternating minimization has found much appeal in areas such as matrix factorization BID9 ), to the best of our knowledge, this is the first such effort in using alternating principles to train neural networks with related performance guarantees.

In this section, we will first set notation and establish the problem setting, then present details of the DANTE method, including the SNGD algorithm.

For sake of simplicity, we consider networks with just a single hidden layer.

We then offer some theoretical insight intro DANTE's inner workings, which also allow us to arrive at the generalized ReLU activation function, and finally describe how DANTE can be extended to deep networks with multiple hidden layers.

Consider a neural network with L layers.

Each layer l ??? {1, 2, . . .

, L} has n l nodes and is characterized by a linear operator W l ??? R n l???1 ??n l and a non-linear activation function ?? l : R n l ??? R n l .

The activations generated by the layer l are denoted by a l ??? R n l .

We denote by a 0 , the input activations and n 0 to be the number of input activations i.e. a 0 ??? R n0 .

Each layer uses activations being fed into it to compute its own activations as a l = ?? l W l , a l???1 ??? R n l , where ?? ., . denotes ??( ., . ) for simplicity of notation.

A multi-layer neural network is formed by nesting such layers to form a composite function f given as follows: DISPLAYFORM0 where W = {W l } is the collection of all the weights through the network, and x = a 0 contains the input activations for each training sample.

Given m data samples {( DISPLAYFORM1 from some distribution D, the network is trained by tuning the weights W to minimize a given loss function, J: DISPLAYFORM2 Note that a multi-layer autoencoder is trained similarly, but with the loss function modified as below: DISPLAYFORM3 For purpose of simplicity and convenience, we first consider the case of a single-layer autoencoder, represented as f (W; x) = ?? 2 W 2 , ?? 1 W 1 , x to describe our methodology.

We describe in a later section on how this idea can be extended to deep multi-layer autoencoders. (Note that our definition of a single-layer autoencoder is equivalent to a two-layer neural network in a classification setting, by nature of the autoencoder.)A common loss function used to train autoencoders is the squared loss function which, in our simplified setting, yields the following objective.

DISPLAYFORM4 An important observation here is that if we fix W 1 , then Eqn (5) turns into a set of Generalized Linear Model problems with ?? 2 as the activation function, i.e. DISPLAYFORM5 where z = ?? 1 W 1 , x .

We exploit this observation in this work.

In particular, we leverage a recent result by BID6 that shows that GLMs with nice, differentiable link functions such as sigmoid (or even a combination of sigmoids such as ?? W2 (??)), satisfy a property the authors name Strict Locally Quasi-Convexity (SLQC), which allows techniques such as SNGD to solve the GLM problems effectively.

Similarly, fixing W 2 turns the problem into yet another SLQC problem, this time with W 1 as the parameter (note that DISPLAYFORM6 //Select a random mini-batch of training points DISPLAYFORM7 Output :Model given by w T This is quite advantageous for us since it allows us to solve each sub-problem of the alternating setup efficiently.

In a subsequent section, we will show that GLMs with non-differentiable activation -in particular, a generalized Rectified Linear Unit (ReLU) -can also satisfy the SLQC property, thus allowing us to extend the proposed alternating strategy, DANTE, to ReLU-based autoencoders too.

We note that while we have developed this idea to train autoencoders in this work (since our approach relates closely to the greedy layer-wise training in autoencoders), DANTE can be used to train standard multi-layer neural networks too (discussed in Section 5).

We begin our presentation of the proposed method by briefly reviewing the Stochastic Normalized Gradient Descent (SNGD) method, which is used to execute the inner steps of DANTE.

We explain in the next subsection, the rationale behind the choice of SNGD as the optimizer.

We stress that although DANTE does use stochastic gradient-style methods internally (such as the SNGD algorithm), the overall strategy adopted by DANTE is not a descent-based strategy, rather an alternating-minimization strategy.

Stochastic Normalized Gradient Descent (SNGD): Normalized Gradient Descent (NGD) is an adaptation of traditional Gradient Descent where the updates in each iteration are purely based on the direction of the gradients, while ignoring their magnitudes.

This is achieved by normalizing the gradients.

SNGD is the stochastic version of NGD, where weight updates are performed using individual (randomly chosen) training samples, instead of the complete set of samples.

Mini-batch SNGD generalizes this by applying updates to the parameters at the end of every mini-batch of samples, as does mini-batch Stochastic Gradient Descent (SGD).

In the remainder of this paper, we refer to mini-batch SNGD as SNGD itself, as is common for SGD.

Algorithm 1 describes the SNGD methodology for a generic GLM problem.

DANTE: Given this background, Algorithm 2 outlines the proposed method, DANTE.

Consider the autoencoder problem below for a single hidden layer network: DISPLAYFORM0 Upon fixing the parameters of the lower layer i.e. W 1 , it is easy to see that we are left with a set of GLM problems: min DISPLAYFORM1 where z = ?? 1 W 1 , x .

DANTE solves this intermediate problem using SNGD steps by sampling several mini-batches of data points and performing updates as dictated by Algorithm 1.

Similarly, fixing the parameters of the upper layer, i.e. W 2 , we are left with another set of problems: DISPLAYFORM2 where ?? W2 ?? = ?? 2 W 2 , ?? 1 ?? .

This is once again solved by mini-batch SNGD, as before.

DISPLAYFORM3 To describe the motivation for our alternating strategy in DANTE, we first define key terms and results that are essential to our work.

We present the notion of a locally quasi-convex function (as introduced in BID6 ) and show that under certain realizability conditions, empirical objective functions induced by Generalized Linear Models (GLMs) are locally quasi-convex.

We then introduce a new activation function, the generalized ReLU, and show that the GLM with the generalized ReLU also satisfies this property.

We cite a result that shows that SNGD converges to the optimum solution provably for locally quasi-convex functions, and subsequently extend this result to the newly introduced activation function.

We also generalize the definition of locally quasi-convex to functions on matrices, which allows us to relate these ideas to layers in neural networks.

DISPLAYFORM4 at least one of the following applies: DISPLAYFORM5 where B (z, /??) refers to a ball centered at z with radius /??.

We generalize this definition to functions on matrices in Appendix A.3.

Definition 3.2 (Idealized and Noisy Generalized Linear Model (GLM)).

Given an (unknown) distribution D and an activation function ?? : R ??? R, an idealized GLM is defined by the existence of a w DISPLAYFORM6 where w * is the global minimizer of the error function: DISPLAYFORM7 Similarly, a noisy GLM is defined by the existence of a w DISPLAYFORM8 , which is the global minimizer of the error function: DISPLAYFORM9 Without any loss in generality, we use x i ??? B d , the unit d-dimensional ball. (Hazan et al., 2015, Lemma 3.2) shows that if we draw m ??? ??? exp(2 w * ) DISPLAYFORM10 from a GLM with the sigmoid activation function, then with probability at least 1 ??? ??, the empirical error function DISPLAYFORM11 However, this result is restrictive, since its proof relies on properties of the sigmoid function, which are not satisfied by other popular activation functions such as the ReLU.

We hence introduce a new generalized ReLU activation function to study the relevance of this result in a broader setting (which has more use in practice).

Definition 3.3. (Generalized ReLU) The generalized ReLU function f : R ??? R, 0 < a < b, a, b ??? R is defined as: DISPLAYFORM12 This function is differentiable at every point except 0.

Note that this definition subsumes variants of ReLU such as the leaky ReLU BID15 ).

We define the function g that provides a valid subgradient for the generalized ReLU at all x to be: DISPLAYFORM13 While SLQC is originally defined for differentiable functions, we now show that with the above definition of the subgradient, the GLM with the generalized ReLU is also SLQC.

This allows us to use the SNGD as an effective optimizer for DANTE to train autoencoders with different kinds of activation functions.

Theorem 3.4.

In the idealized GLM with generalized ReLU activation, assuming ||w DISPLAYFORM14 where m is the total number of samples.

Also let v be a point /??-close to minima w * with ?? = 2b 3 W a .

Let g be the subgradient of the generalized ReLU activation and G be the subgradient of?? rr m (w). (Note that as before, g ., . denotes g( ., . )).

Then: DISPLAYFORM15 In the above proof, we first use the fact (in Step 1) that in the GLM, there is some w * such that ?? w * , x i = y i .

Then, we use the fact (in Steps 2 and 4) that the generalized ReLU function is b-Lipschitz, and the fact that the minimum value of the quasigradient of g is a (Step 3).

Subsequently, inStep 5, we simply use the given bounds on the variables x i , w, w * due to the setup of the problem (w ??? B d (0, W ), and x i ??? B d , the unit d-dimensional ball, as defined earlier in this section).We also prove a similar result for the Noisy GLM below.

Theorem 3.5.

In the noisy GLM with generalized ReLU activation, assuming ||w * || ??? W , given w ??? B(0, W ), then with probability DISPLAYFORM16 The proof for Theorem 3.5 is included in Appendix A.1.We connect the above results with a result from BID6 (stated below) which shows that SNGD provably converges to the optimum for SLQC functions, and hence, with very high probability, for empirical objective functions induced by noisy GLM instances too.

Theorem 3.6 BID6 ).

Let , ??, G, M, ?? > 0, let f : R d ??? R and w * = arg min w f (w).

Assume that for b ??? b 0 ( , ??, T ), with probability ??? 1 ??? ??, f t defined in Algorithm 1 is ( , ??, w * )-SLQC ???w, and |f t | ??? M ???t ??? {1, ?? ?? ?? , T } .

If we run SNGD with T ??? DISPLAYFORM17 and ?? = ?? , and b ??? max DISPLAYFORM18 The results so far show that SNGD provides provable convergence for idealized and noisy GLM problems with both sigmoid and ReLU family of activation functions.

We note that alternate activation functions such as tanh (which is simply a rescaled sigmoid) and leaky ReLU BID15 ) are variants of the aforementioned functions.

In Algorithm 2, it is evident that each node of the output layer presents a GLM problem (and hence, SLQC) w.r.t.

the corresponding weights from W 2 .

We show in Appendices A.2 and A.3 how the entire layer is SLQC w.r.t.

W 2 , by generalizing the definition of SLQC to matrices.

In case of W 1 , while the problem may not directly represent a GLM, we show in Appendix A.3 that our generalized definition of SLQC to functions on matrices allows us to prove that Step 4 of Algorithm 2 is also SLQC w.r.t.

W 1 .Thus, given a single-layer autoencoder with either sigmoid or ReLU activation functions, DANTE provides an effective alternating minimization strategy that uses SNGD to solve SLQC problems in each alternating step, each of which converges to its respective -suboptimal solution with high probability, as shown above in Theorem 3.6.

Importantly, note that the convergence rate of SNGD depends on the ?? parameter.

Whereas the GLM error function with sigmoid activation has ?? = e W Hazan et al. FORMULA0 , we obtain ?? = 2b 3 W a (i.e. linear in W ) for the generalized ReLU setting, which is an exponential improvement.

This is significant as in Theorem 3.6, the number of iterations T depends on ?? 2 .

This shows that SNGD offers accelerated convergence with generalized ReLU GLMs (introduced in this work) when compared to sigmoid GLMs.

In the previous sections, we illustrated how a single hidden-layer autoencoder can be cast as a set of SLQC problems and proposed an alternating minimization method, DANTE.

This approach can be generalized to deep autoencoders by considering the greedy layer-wise approach to training a neural network BID1 ).

In this approach, each pair of layers of a deep stacked autoencoder is successively trained in order to obtain the final representation.

Each pair of layers considered in this paradigm is a single hidden-layer autoencoder, which can be cast as pairs of SLQC problems that can be trained using DANTE.

Therefore, training a deep autoencoder using greedy layer-wise approach can be modeled as a series of SLQC problem pairs.

Algorithm 3 summarizes the proposed approach to use DANTE for a deep autoencoder, and Figure 1 illustrates the approach.

Note that it may be possible to use other schemes to use DANTE for multi-layer autoencoders such as a round-robin scheme, where each layer is trained separately one after the other in the sequence in which the layers appear in the network.

We validated DANTE by training autoencoders on an expanded 32??32 variant of the standard MNIST dataset BID10 ) as well as other datasets from the UCI repository.

We also conducted experiments with multi-layer autoencoders, as well as studied with varying number of hidden neurons Figure 1: An illustration of the proposed multi-layer DANTE (best viewed in color).

In each training phase, the outer pairs of weights (shaded in gold) are treated as a single-layer autoencoder to be trained using single-layer DANTE, followed by the inner single-layer auroencoder (shaded in black).

These two phases are followed by a finetuning process that may be empirically determined, similar to standard deep autoencoder training.

Algorithm 3: DANTE for a multi-layer autoencoder Input :Encoder e with weights U, Decoder d with weights V, Number of hidden layers 2n ??? 1, Learning rate ??, Stopping threshold , Number of iterations of alternating minimization Output :U, V on single-layer autoencoders.

Our experiments on MNIST used the standard benchmarking setup of the dataset 1 , with 60, 000 data samples used for training and 10, 000 samples for testing.

Experiments were conducted using Torch 7 BID5 ).

DISPLAYFORM0 Autoencoder with Sigmoid Activation: A single-layer autoencoder (equivalent to a neural network with one hidden layer) with a sigmoid activation was trained using DANTE as well as standard backprop-SGD (represented as SGD in the results, for convenience) using the standard Mean-Squared Error loss function.

The experiments considered 600 hidden units, a learning rate of 0.001, and a minibatch size of 500 (same setup was maintained for SGD and the SNGD used inside DANTE for fair comparison; one could optimize both SGD and SNGD to improve the absolute result values.)

We studied the performance by varying the number of hidden neurons, and show those results later in this section.

The results are shown in FIG2 .

The figure shows that while DANTE takes slightly (negligibly) longer to reach a local minimum, it obtains a better solution than SGD. (We note that the time taken for the iterations were comparable across both DANTE and backprop-SGD.)Autoencoder with ReLU Activation: Similar to the above experiment, a single-layer autoencoder with a leaky ReLU activation was trained using DANTE and backprop-SGD using the Mean-Squared Error loss function.

Once again, the experiments considered 600 units in the hidden layer of the autoencoder, a leakiness parameter of 0.01 for the leaky ReLU, a learning rate of 0.001, and a minibatch size of 500.

The results are shown in FIG2 .

The results for ReLU showed an improvement, and DANTE was marginally better than back-prop SGD across the iterations (as shown in the figure) .In FIG3 , we also show the reconstructions obtained by both trained models (DANTE and Backprop-SGD) for the autoencoder with the Generalized ReLU activation.

The model trained using DANTE shows comparable performance as a model trained by SGD under the same settings, in this case.

We also conducted experiments to study the effectiveness of the feature representations learned using the models trained using DANTE and SGD in the same setting.

After training, we passed the dataset through the autoencoder, extracted the hidden layer representations, and then trained a linear SVM.

The classification accuracy results using the hidden representations are given in Table 1.

The table clearly shows the competitive performance of DANTE on this task.

We also studied the performance of DANTE on other standard datasets 2 , viz.

Ionosphere (34 dimensions, 351 datapoints), SVMGuide4 (10 dimensions, 300 datapoints), Vehicle (18 dimensions, 846 datapoints), and USPS (256 dimensions, 7291 datapoints).

Table 1 show the performance of the proposed method vs SGD on the abovementioned datasets.

It can be seen that DANTE once again demonstrates competitive performance across the datasets, presenting its capability as a viable alternative for standard backprop-SGD.Varying Number of Hidden Neurons: Given the decomposable nature of the proposed solution to learning autoencoders, we also studied the effect of varying hyperparameters across the layers, in particular, the number of hidden neurons in a single-layer autoencoder.

The results of these experiments are shown in Figure 5 .

The plots show that when the number of hidden neurons is low, DANTE reaches its minumum value much sooner (considering this is a subgradient method, one can always choose the best iterate over training) than SGD, although SGD finds a slightly better solution.

However, when the number of hidden neurons increases, DANTE starts getting consistently better.

This can be attributed to the fact that the subproblem is relatively more challenging for an alternating optimization setting when the number of hidden neurons is lesser.(a) Architecture:

1024->500->500->1024 (b) Architecture: 1024->750->500->750->1024 Figure 6 : Plots of training error and test error vs training iterations for multi-layer autoencoders with generalized (leaky) ReLU activations for both DANTE and SGD.Multi-Layer Autoencoder: We also studied the performance of the proposed multi-layer DANTE method (Algorithm 3) for the MNIST dataset.

Figure 6 shows the results obtained by stacking two single-layer autoencoders, each with the generalized (leaky) ReLU activation (note that a two singlelayer autoencoder corresponds to 4 layers in the overall network, as mentioned in the architecture on the figure) .

The figure shows promising performance for DANTE in this experiment.

Note that Figure 6b shows two spikes: one when the training for the next pair of layers in the autoencoder begins, and another when the end-to-end finetuning process is done.

This is not present in Figure 6a , since the 500 ??? 500 layer in between is only randomly initialized, and is not trained using DANTE or SGD.

In this work, we presented a novel methodology, Deep AlterNations for Training autoEncoders (DANTE), to efficiently train autoencoders using alternating minimization, thus providing an effective alternative to backpropagation.

We formulated the task of training each layer of an autoencoder as a Strictly Locally Quasi-Convex (SLQC) problem, and leveraged recent results to use Stochastic Normalized Gradient Descent (SNGD) as an effective method to train each layer of the autoencoder.

While recent work was restricted to using sigmoidal activation functions, we introduced a new generalized ReLU activation function, and showed that a GLM with this activation function also satisfies the SLQC property, thus allowing us to expand the applicability of the proposed method to autoencoders with both sigmoid and ReLU family of activation functions.

In particular, we extended the definitions of local quasi-convexity to use subgradients in order to prove that the GLM with generalized ReLU activation is , DISPLAYFORM0 , w * ??? SLQC, which improves the convergence bound for SLQC in the GLM with the generalized ReLU (as compared to a GLM with sigmoid).

We also showed how DANTE can be extended to train multi-layer autoencoders.

We empirically validated DANTE with both sigmoidal and ReLU activations on standard datasets as well as in a multi-layer setting, and observed that it provides a competitive alternative to standard backprop-SGD, as evidenced in the experimental results.

Future Work and Extensions.

DANTE can not only be used to train autoencoders, but can be extended to train standard multi-layer neural networks too.

One could use DANTE to train a neural network layer-wise in a round robin fashion, and then finetune end-to-end using backprop-SGD.

In case of autoencoders with tied weights, one could use DANTE to learn the weights of the required layers, and then finetune end-to-end using a method such as SGD.

Our future work will involve a more careful study of the proposed method for deeper autoencoders, including the settings mentioned above, as well as in studying performance bounds for the end-to-end alternating minimization strategy for the proposed method.

The theorem below is a continuation of the discussion in Section 3.3 (see Theorem 3.5).

We prove this result below.

Theorem A.1.

In the noisy GLM with generalized ReLU activation, assuming ||w * || ??? W , given w ??? B(0, W ), then with probability DISPLAYFORM0 Proof.

Here, ???i, y i ??? [0, 1], the following holds: DISPLAYFORM1 where DISPLAYFORM2 are zero mean, independent and bounded random variables, i.e. ???i ??? [m], ||?? i || ??? 1.

Then,?? rr m (w) may be written as follows (expanding y i as in Eqn 6): DISPLAYFORM3 Therefore, we also have (by definition of noisy GLM in Defn 3.2): DISPLAYFORM4 Consider ||w|| ??? W such that?? rr m (w) ????? rr m (w * ) ??? .

Also, let v be a point /??-close to minima w * with ?? = 2b 3 W a .

Let g be the subgradient of the generalized ReLU activation and G be the subgradient of?? rr m (w), as before.

Then: DISPLAYFORM5 Here, ?? i (w) = 2g w, DISPLAYFORM6 The above proof uses arguments similar to the proof for the idealized GLM (please see the lines after the proof of Theorem 3.4, viz.

the b-Lipschitzness of the generalized ReLU, and the problem setup).

Now, when 1 m DISPLAYFORM7 our model is SLQC.

By simply using the Hoeffding's bound, we get that the theorem statement holds for m ??? DISPLAYFORM8

Given an (unknown) distribution D, let the layer be characterized by a linear operator W ??? R d??d and a non-linear activation function defined by ?? : R ??? R. Let the layer output be defined by ?? W, x , where x ??? R d is the input, and ?? is used element-wise in this function.

Consider the mean squared error loss, commonly used in autoencoders, given by: min Each of these sub-problems above is a GLM, which can be solved effectively using SNGD as seen in Theorem 3.6, which we leverage in this work.

In Algorithm 2, while it is evident that each of the problems in Step 3 is a GLM and hence, SLQC, w.r.t.

the corresponding parameters in W 2 , we show here that the complete layer in Step 3 is also SLQC w.r.t.

W 2 , as well as show that the problem in Step 4 is SLQC w.r.t.

W 1 .

We begin with the definition of SLQC for matrices, which is defined using the Frobenius inner product.

Definition A.2 (Local-Quasi-Convexity for Matrices).

Let x, z ??? R d??d , ??, > 0 and let f : R d??d ??? R be a differentiable function.

Then f is said to be ( , ??, z)-Strictly-Locally-Quasi-Convex (SLQC) in x, if at least one of the following applies:1.

f (x) ??? f (z) ??? 2.

???f (x) > 0, and ???y ??? B (z, /??), T r(???f (x)T (y ??? x)) ??? 0 where B (z, /??) refers to a ball centered at z with radius /??.

We now prove that the?? rr(W) of a multi-output single-layer neural network is indeed SLQC in W. This corresponds to proving that the one-hidden layer autoencoder problem is SLQC in W 2 .

We then go on to prove that a two layer single-output neural network is SLQC in the first layer W 1 , which can be trivially extended using the basic idea seen in Theorem A.4 to show that the one hidden-layer autoencoder problem is also SLQC in W 1 .

Theorem A.3.

Let an idealized single-layer multi-output neural network be characterized by a linear operator W ??? R d??d = [w 1 w 2 ?? ?? ?? w d ] and a generalized ReLU activation function ?? : R ??? R. Let the output of the layer be ?? W, x where x ??? R d is the input, and ?? is applied element-wise.

Assuming ||W * || ??? C,?? rr(W) is , DISPLAYFORM0 The remainder of the proof proceeds precisely as in Theorem 3.4.Theorem A.4.

Let an idealized two-layer neural network be characterized by a linear operator w 1 ??? R d??d , w 2 ??? R d and generalized ReLU activation functions ?? 1 : R d ??? R d , ?? 2 : R ??? R with a setting similar to Equation 5.

Assuming ||w

<|TLDR|>

@highlight

We utilize the alternating minimization principle to provide an effective novel technique to train deep autoencoders.

@highlight

Alternating minimization framework for training autoencoder and encoder-decoder networks

@highlight

The authors explore an alternating optimization approach for training Auto Encoders, treating each layer as a generalized linear model, and suggest using the stochastic normalized GD as the minimization algorithm in each phase.