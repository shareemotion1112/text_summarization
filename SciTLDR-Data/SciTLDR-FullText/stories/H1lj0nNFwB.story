A leading hypothesis for the surprising generalization of neural networks is that the dynamics of gradient descent bias the model towards simple solutions, by searching through the solution space in an incremental order of complexity.

We formally define the notion of incremental learning dynamics and derive the conditions on depth and initialization for which this phenomenon arises in deep linear models.

Our main theoretical contribution is a dynamical depth separation result, proving that while shallow models can exhibit incremental learning dynamics, they require the initialization to be exponentially small for these dynamics to present themselves.

However, once the model becomes deeper, the dependence becomes polynomial and incremental learning can arise in more natural settings.

We complement our theoretical findings by experimenting with deep matrix sensing, quadratic neural networks and with binary classification using diagonal and convolutional linear networks, showing all of these models exhibit incremental learning.

Neural networks have led to a breakthrough in modern machine learning, allowing us to efficiently learn highly expressive models that still generalize to unseen data.

The theoretical reasons for this success are still unclear, as the generalization capabilities of neural networks defy the classic statistical learning theory bounds.

Since these bounds, which depend solely on the capacity of the learned model, are unable to account for the success of neural networks, we must examine additional properties of the learning process.

One such property is the optimization algorithm -while neural networks can express a multitude of possible ERM solutions for a given training set, gradient-based methods with the right initialization may be implicitly biased towards certain solutions which generalize.

A possible way such an implicit bias may present itself, is if gradient-based methods were to search the hypothesis space for possible solutions of gradually increasing complexity.

This would suggest that while the hypothesis space itself is extremely complex, our search strategy favors the simplest solutions and thus generalizes.

One of the leading results along these lines has been by Saxe et al. (2013) , deriving an analytical solution for the gradient flow dynamics of deep linear networks and showing that for such models, the singular values converge at different rates, with larger values converging first.

At the limit of infinitesimal initialization of the deep linear network, Gidel et al. (2019) show these dynamics exhibit a behavior of "incremental learning" -the singular values of the model are learned separately, one at a time.

Our work generalizes these results to small but finite initialization scales.

Incremental learning dynamics have also been explored in gradient descent applied to matrix completion and sensing with a factorized parameterization (Gunasekar et al. (2017) , Arora et al. (2018) , Woodworth et al. (2019) ).

When initialized with small Gaussian weights and trained with a small learning rate, such a model is able to successfully recover the low-rank matrix which labeled the data, even if the problem is highly over-determined and no additional regularization is applied.

In their proof of low-rank recovery for such models, Li et al. (2017) show that the model remains lowrank throughout the optimization process, leading to the successful generalization.

Additionally, Arora et al. (2019) explore the dynamics of such models, showing the singular values are learned at different rates and that deeper models exhibit stronger incremental learning dynamics.

Our work deals with a more simplified setting, allowing us to determine explicitly under which conditions depth leads to this dynamical phenomenon.

Finally, the learning dynamics of nonlinear models have been studied as well.

Combes et al. (2018) and Williams et al. (2019) study the gradient flow dynamics of shallow ReLU networks under restrictive distributional assumptions, Ronen et al. (2019) show that shallow networks learn functions of gradually increasing frequencies and Nakkiran et al. (2019) show how deep ReLU networks correlate with linear classifiers in the early stages of training.

These findings, along with others, suggest that the generalization ability of deep networks is at least in part due to the incremental learning dynamics of gradient descent.

Following this line of work, we begin by explicitly defining the notion of incremental learning for a toy model which exhibits this sort of behavior.

Analyzing the dynamics of the model for gradient flow and gradient descent, we characterize the effect of the model's depth and initialization scale on incremental learning, showing how deeper models allow for incremental learning in larger (realistic) initialization scales.

Specifically, we show that a depth-2 model requires exponentially small initialization for incremental learning to occur, while deeper models only require the initialization to be polynomially small.

Once incremental learning has been defined and characterized for the toy model, we generalize our results theoretically and empirically for larger linear and quadratic models.

Examples of incremental learning in these models can be seen in figure 1, which we discuss further in section 4.

We begin by analyzing incremental learning for a simple model.

This will allow us to gain a clear understanding of the phenomenon and the conditions for it, which we will later be able to apply to a variety of other models in which incremental learning is present.

Our simple linear model will be similar to the toy model analyzed by Woodworth et al. (2019) .

Our input space will be X = R d and the hypothesis space will be linear models with non-negative weights, such that:

We will introduce depth into our model, by parameterizing ?? using w ??? R d ???0 in the following way:

Where N represents the depth of the model.

Since we restrict the model to having non-negative weights, this parameterization doesn't change the expressiveness, but it does radically change it's optimization dynamics.

Assuming the data is labeled by some ?? * ??? R d ???0 , we will study the dynamics of this model for general N under a depth-normalized 1 squared loss over Gaussian inputs, which will allow us to derive our analytical solution:

We will assume that our model is initialized uniformly with a tunable scaling factor, such that:

2.2 GRADIENT FLOW ANALYTICAL SOLUTIONS Analyzing our toy model using gradient flow allows us to obtain an analytical solution for the dynamics of ??(t) along with the dynamics of the loss function for a general N .

For brevity, the following theorem refers only to N = 1, 2 and N ??? ???, however the solutions for 3 ??? N < ??? are similar in structure to N ??? ???, but more complicated.

We also assume ?? * i > 0 for brevity, however we can derive the solutions for ?? * i = 0 as well.

Note that this result is a special case adaptation of the one presented in Saxe et al. (2013) for deep linear networks: Theorem 1.

Minimizing the toy linear model described in (1) with gradient flow over the depth normalized squared loss (2), with Gaussian inputs and weights initialized as in (3) and assuming ?? * i > 0 leads to the following analytical solutions for different values of N :

Proof.

The gradient flow equations for our model are the following:

Given the dynamics of the w parameters, we may use the chain rule to derive the dynamics of the induced model, ??:??

This differential equation is solvable for all N , leading to the solutions in the theorem.

Taking

, which is also solvable.

for ?? * i ??? {12, 6, 4, 3} according to the analytical solutions in theorem 1, under different depths and initializations.

The first column has all values converging at the same rate.

Notice how the deep parameterization with small initialization leads to distinct phases of learning, where values are learned incrementally (bottom-right).

The shallow model's much weaker incremental learning, even at small initialization scales (second column), is explained in theorem 2.

Analyzing these solutions, we see how even in such a simple model depth causes different factors of the model to be learned at different rates.

Specifically, values corresponding to larger optimal values converge faster, suggesting a form of incremental learning.

This is most clear for N = 2 where the solution isn't implicit, but is also the case for N ??? 3, as we will see in the next subsection.

These dynamics are depicted in figure 2, where we see the dynamics of the different values of ??(t) as learning progresses.

When N = 1, all values are learned at the same rate regardless of the initialization, while the deeper models are clearly biased towards learning the larger singular values first, especially at small initialization scales.

Our model has only one optimal solution due to the population loss, but it is clear how this sort of dynamic can induce sparse solutions -if the model is able to fit the data after a small amount of learning phases, then it's obtained result will be sparse.

Alternatively, if N = 1, we know that the dynamics will lead to the minimal 2 norm solution which is dense.

We explore the sparsity inducing bias of our toy model by comparing it empirically 2 to a greedy sparse approximation algorithm in appendix D, and give our theoretical results in the next section.

Equipped with analytical solutions for the dynamics of our model for every depth, we turn to study how the depth and initialization effect incremental learning.

While Gidel et al. (2019) focuses on incremental learning in depth-2 models at the limit of ?? 0 ??? 0, we will study the phenomenon for a general depth and for ?? 0 > 0.

First, we will define the notion of incremental learning.

Since all values of ?? are learned in parallel, we can't expect one value to converge before the other moves at all (which happens for infinitesimal initialization as shown by Gidel et al. (2019) ).

We will need a more relaxed definition for incremental learning in finite initialization scales.

Definition 1.

Given two values ?? i , ?? j such that ?? * i > ?? * j > 0 and both are initialized as ?? i (0) = ?? j (0) = ?? 0 < ?? * j , and given two scalars s ??? (0, 1 4 ) and f ??? ( 3 4 , 1), we call the learning of the values (s, f )-incremental if there exists a t for which:

In words, two values have distinct learning phases if the first almost converges (f ??? 1) before the second changes by much (s 1).

Note that for any N , ??(t) is monotonically increasing and so once ?? j (t) = s?? * j , it will not decrease to allow further incremental learning.

Given this definition of incremental learning, we turn to study the conditions that facilitate incremental learning in our toy model.

Our main result is a dynamical depth separation result, showing that incremental learning is dependent on

Proof sketch (the full proof is given in appendix A).

Rewriting the separable differential equation in (4) to calculate the time until ??(t) = ???? * , we get the following:

The condition for incremental learning is then the requirement that t f (?? i ) ??? t s (?? j ), resulting in:

We then relax/restrict the above condition to get a necessary/sufficient condition on ?? 0 , leading to a lower and upper bound on ?? th 0 .

Note that the value determining the condition for incremental learning is

-if two values are in the same order of magnitude, then their ratio will be close to 1 and we will need a small initialization to obtain incremental learning.

The dependence on the ratio changes with depth, and is exponential for N = 2.

This means that incremental learning, while possible for shallow models, is difficult to see in practice.

This result explains why changing the initialization scale in figure 2 changes the dynamics of the N ??? 3 models, while not changing the dynamics for N = 2 noticeably.

The next theorem extends part of our analysis to gradient descent, a more realistic setting than the infinitesimal learning rate of gradient flow:

Theorem 3.

Given two values ?? i , ?? j of a depth-2 toy linear model as in (1), such that

and the model is initialized as in (3), and given two scalars s ??? (0, 1 4 ) and f ??? ( 3 4 , 1), and assuming ?? * j ??? 2?? 0 , and assuming we optimize with gradient descent with a learning rate ?? ??? c ?? * c < 2( ??? 2 ??? 1) and ?? * 1 the largest value of ?? * , then the largest initialization value for which the learning phases of the values are (s, f )-incremental, denoted ?? th 0 , is lower and upper bounded in the following way:

Where A and B are defined as:

We defer the proof to appendix B.

Note that this result, while less elegant than the bounds of the gradient flow analysis, is similar in nature.

Both A and B simplify to r when we take their first order approximation around c = 0, giving us similar bounds and showing that the condition on ?? 0 for N = 2 is exponential in gradient descent as well.

While similar gradient descent results are harder to obtain for deeper models, we discuss the general effect of depth on the gradient decent dynamics in appendix C.

So far, we have only shown interesting properties of incremental learning caused by depth for a toy model.

In this section, we will relate several deep models to our toy model and show how incremental learning presents itself in larger models as well.

The task of matrix sensing is a generalization of matrix completion, where our input space is X = R d??d and our model is a matrix W ??? R d??d , such that:

Following Arora et al. (2019), we introduce depth by parameterizing the model using a product of matrices and the following initialization scheme (W i ??? R d??d ):

Note that when d = 1, the deep matrix sensing model reduces to our toy model without weight sharing.

We study the dynamics of the model under gradient flow over a depth-normalized squared loss, assuming the data is labeled by a matrix sensing model parameterized by a PSD W * ??? R d??d :

The following theorem relates the deep matrix sensing model to our toy model, showing the two have the same dynamical equations:

Theorem 4.

Optimizing the deep matrix sensing model described in (5) with gradient flow over the depth normalized squared loss ((6)), with weights initialized as in (5) leads to the following dynamical equations for different values of N :

Where ?? i and ?? * i are the ith singular values of W and W * , respectively, corresponding to the same singular vector.

The proof follows that of Saxe et al. (2013) and Gidel et al. (2019) and is deferred to appendix E.

Theorem 4 shows us that the bias towards sparse solutions introduced by depth in the toy model is equivalent to the bias for low-rank solutions in the matrix sensing task.

This bias was studied in a more general setting in Arora et al. (2019) , with empirical results supporting the effect of depth on the obtainment of low-rank solutions under a more natural loss and initialization scheme.

We recreate and discuss these experiments and their connection to our analysis in appendix E, and an example of these dynamics in deep matrix sensing can also be seen in panel (a) of figure 1.

By drawing connections between quadratic networks and matrix sensing (as in Soltanolkotabi et al. (2018)), we can extend our results to these nonlinear models.

We will study a simplified quadratic network, where our input space is X = R d and the first layer is parameterized by a weight matrix W ??? R d??d and followed by a quadratic activation function.

The final layer will be a summation layer.

We assume, like before, that the labeling function is a quadratic network parameterized by W * ??? R d??d .

Our model can be written in the following way, using the following orthogonal initialization scheme:

Immediately, we see the similarity of the quadratic network to the deep matrix sensing model with N = 2, where the input space is made up of rank-1 matrices.

However, the change in input space forces us to optimize over a different loss function to reproduce the same dynamics: Definition 2.

Given an input distribution over an input space X with a labeling function y : X ??? R and a hypothesis h, the variance loss is defined in the following way:

Note that minimizing this loss function amounts to minimizing the variance of the error, while the squared loss minimizes the second moment of the error.

We note that both loss functions have the same minimum for our problem, and the dynamics of the squared loss can be approximated in certain cases by the dynamics of the variance loss.

For a complete discussion of the two losses, including the cases where the two losses have similar dynamics, we refer the reader to appendix F. Theorem 5.

Minimizing the quadratic network described and initialized as in (7) with gradient flow over the variance loss defined in (2) leads to the following dynamical equations:

Where ?? i and ?? * i are the ith singular values of W and W * , respectively, corresponding to the same singular vector.

We defer the proof to appendix F and note that these dynamics are the same as our depth-2 toy model, showing that shallow quadratic networks can exhibit incremental learning (albeit requiring a small initialization).

While incremental learning has been described for deep linear networks in the past, it has been restricted to regression tasks.

Here, we illustrate how incremental learning presents itself in binary classification, where implicit bias results have so far focused on convergence at t ??? ??? (Soudry et al. (2018), Nacson et al. (2018) , Ji & Telgarsky (2019) ).

Deep linear networks with diagonal weight matrices have been shown to be biased towards sparse solutions when N > 1 in Gunasekar et al. (2018) , and biased towards the max-margin solution for N = 1.

Instead of analyzing convergence at t ??? ???, we intend to show that the model favors sparse solutions for the entire duration of optimization, and that this is due to the dynamics of incremental learning.

Our theoretical illustration will use our toy model as in (1) (initialized as in (3)) as a special weightshared case of deep networks with diagonal weight matrices, and we will then show empirical results for the more general setting.

We analyze the optimization dynamics of this model over a separable

where y i ??? {??1}. We use the exponential loss ( (f (x), y) = e ???yf (x) ) for the theoretical illustration and experiment on the exponential and logistic losses.

Computing the gradient for the model over w, the gradient flow dynamics for ?? become:

We see the same dynamical attenuation of small values of ?? that is seen in the regression model, caused by the multiplication by ?? .

From this, we can expect the same type of incremental learning to occur -weights of ?? will be learned incrementally until the dataset can be separated by the current support of ??.

Then, the dynamics strengthen the growth of the current support while relatively attenuating that of the other values.

Since the data is separated, increasing the values of the current support reduces the loss and the magnitude of subsequent gradients, and so we should expect the support to remain the same and the model to converge to a sparse solution.

Granted, the above description is just intuition, but panel (c) of figure 1 shows how it is born out in practice (similar results are obtained for the logistic loss).

In appendix G we further explore this model, showing deeper networks have a stronger bias for sparsity.

We also observe that the initialization scale plays a similar role as before -deep models are less biased towards sparsity when ?? 0 is large.

In their work, Gunasekar et al. (2018) show an equivalence between the diagonal network and the circular-convolutional network in the frequency domain.

According to their results, we should expect to see the same sparsity-bias of diagonal networks in convolutional networks, when looking at the Fourier coefficients of ??.

An example of this can be seen in panel (d) of figure 1, and we refer the reader to appendix G for a full discussion of their convolutional model and it's incremental learning dynamics.

Gradient-based optimization for deep linear models has an implicit bias towards simple (sparse) solutions, caused by an incremental search strategy over the hypothesis space.

Deeper models have a stronger tendency for incremental learning, exhibiting it in more realistic initialization scales.

This dynamical phenomenon exists for the entire optimization process for regression as well as classification tasks, and for many types of models -diagonal networks, convolutional networks, matrix completion and even the nonlinear quadratic network.

We believe this kind of dynamical analysis may be able to shed light on the generalization of deeper nonlinear neural networks as well, with shallow quadratic networks being only a first step towards that goal.

Proof.

Our strategy will be to define the time t ?? for which a value reaches a fraction ?? of it's optimal value, and then require that t f (?? i ) ??? t s (?? j ).

We begin with recalling the differential equation which determines the dynamics of the model:??

Since the solution for N ??? 3 is implicit and difficult to manage in a general form, we will define t ?? using the integral of the differential equation.

The equation is separable, and under initialization of ?? 0 we can describe t ?? (??) in the following way:

Incremental learning takes place when ?? i (t f ) = f ?? * i happens before ?? j (t s ) = s?? * j .

We can write this condition in the following way:

Plugging in ?? i = r?? j and rearranging, we get the following necessary and sufficient condition for incremental learning:

Our last step before relaxing and restricting our condition will be to split the integral on the left-hand side into two integrals:

At this point, we cannot solve this equation and isolate ?? 0 to obtain a clear threshold condition on it for incremental learning.

Instead, we will relax/restrict the above condition to get a necessary/sufficient condition on ?? 0 , leading to a lower and upper bound on the threshold value of ?? 0 .

To obtain a sufficient (but not necessary) condition on ?? 0 , we may make the condition stricter either by increasing the left-hand side or decreasing the right-hand side.

We can increase the left-hand side by removing r from the left-most integral's denominator (r > 1) and then combine the left-most and right-most integrals:

Next, we note that the integration bounds give us a bound on ?? for either integral.

This means we can replace 1 ??? ?? ?? * j with 1 on the right-hand side, and replace 1 ??? ?? r?? * j with 1 ??? f on the left-hand side:

We may now solve these integrals for every N and isolate ?? 0 , obtaining the lower bound on ?? th 0 .

We start with the case where N = 2:

Rearranging to isolate ?? 0 , we obtain our result:

For the N ??? 3 case, we have the following after solving the integrals:

For simplicity we may further restrict the condition by removing the term 1 rf ?? * j 1??? 2 N .

Solving for ?? 0 gives us the following:

To obtain a necessary (but not sufficient) condition on ?? 0 , we may relax the condition in (8) either by decreasing the left-hand side or increasing the right-hand side.

We begin by rearranging the equation:

Like before, we may use the integration bounds to bound ??.

Plugging in ?? = s?? * j for all integrals decreases the left-hand side and increases the right-hand side, leading us to the following:

Rearranging, we get the following inequality:

We now solve the integrals for the different cases.

For N = 2, we have:

Rearranging to isolate ?? 0 , we get our condition:

Finally, for N ??? 3, we solve the integrals to give us:

Rearranging to isolate ?? 0 , we get our condition:

For a given N , we derived a sufficient condition and a necessary condition on ?? 0 for (s, f )-incremental learning.

The necessary and sufficient condition on ?? 0 , which is the largest initialization value for which we see incremental learning (denoted ?? th 0 ), is between the two derived bounds.

The precise bounds can possibly be improved a bit, but the asymptotic dependence on r is the crux of the matter, showing the dependence on r changes with depth with a substantial difference when we move from shallow models (N = 2) to deeper ones (N ??? 3)

Theorem.

Given two values ?? i , ?? j of a depth-2 toy linear model as in (1), such that

and the model is initialized as in (3), and given two scalars s ??? (0, 1 4 ) and f ??? ( 3 4 , 1), and assuming ?? * j ??? 2?? 0 , and assuming we optimize with gradient descent with a learning rate ?? ??? c ?? * 1 for c < 2( ??? 2 ??? 1) and ?? * 1 the largest value of ?? * , then the largest initialization value for which the learning phases of the values are (s, f )-incremental, denoted ?? th 0 , is lower and upper bounded in the following way:

Where A and B are defined as:

Proof.

To show our result for gradient descent and N = 2, we build on the proof techniques of theorem 3 of Gidel et al. (2019) .

We start by deriving the recurrence relation for the values ??(t) for general depth, when t now stands for the iteration.

Remembering that w n i = ?? i , we write down the gradient update for w i (t):

Raising w i (t) to the N th power, we get the gradient update for the ?? values:

Next, we will prove a simple lemma which gives us the maximal learning rate we will consider for the analysis, for which there is no overshooting (the values don't grow larger than the optimal values).

Lemma 1.

For the gradient update in (9), assuming

for c ??? 1, we have:

Defining r i = ??i ?? * i and dividing both sides by ?? * i , we have:

It is enough to show that for any 0 ??? r ??? 1, we have that re r 1??? 2 N (1???r) ??? 1, as over-shooting occurs when r i (t) > 1.

Indeed, this function is monotonic increasing in 0 ??? r ??? 1 (since the exponent is non negative), and equals 1 when r = 1.

Since r = 1 is a fixed point and no iteration that starts at r < 1 can cross 1, then r i (t) ??? 1 for any t. This concludes our proof.

Under this choice of learning rate, we can now obtain our incremental learning results for gradient descent when N = 2.

Our strategy will be bounding ?? i (t) from below and above, which will give us a lower and upper bound for t ?? (?? i ).

Once we have these bounds, we will be able to describe either a necessary or a sufficient condition on ?? 0 for incremental learning, similar to theorem 2.

The update rule for N = 2 is:

Next, we plug in ?? =

Where in the fourth line we use the inequality 1 1+x ??? 1 ??? x, ???x ??? 0.

We may now subtract 1 ?? * i from both sides to obtain:

We may now obtain a bound on t ?? (?? i ) by plugging in ?? i (t) = ???? * i and taking the log:

Rearranging (note that log 1 ??? cR i ??? c 2 4 R 2 i < 0 and that our choice of c keeps the argument of the log positive), we get:

Next, we follow the same procedure for an upper bound.

Starting with our update step:

Where in the last line we use the inequality

from both sides, we get:

Rearranging like before, we get the bound on the ??-time:

Given these bounds, we would like to find the conditions on ?? 0 that allows for (s, f )-incremental learning.

We will find a sufficient condition and a necessary condition, like in the proof of theorem 2.

A sufficient condition for incremental learning will be one which is possibly stricter than the exact condition.

We can find such a condition by requiring the upper bound of t f (?? i ) to be smaller than the lower bound on t s (?? j ).

This becomes the following condition:

and rearranging, we get the following:

We may now take the exponent of both sides and rearrange again, remembering

= r > 1, to get the following condition:

Now, we will add the very reasonable assumption that ?? * j ??? 2?? 0 , which allows us to replace

Now we can rearrange and isolate ?? 0 to get a sufficient condition for incremental learning:

A necessary condition for incremental learning will be one which is possibly more relaxed than the exact condition.

We can find such a condition by requiring the lower bound of t f (?? i ) to be smaller than the upper bound on t s (?? j ).

This becomes the following condition:

log 1???cRj +c 2 R 2 j and rearranging, we get the following:

We may now take the exponent of both sides and rearrange again, remembering ?? * i ?? * j = r > 1, to get the following condition:

We may now relax the condition further, by removing the r from the denominator of the left-hand side and the ?? 0 from the numerator.

This gives us the following:

Finally, rearranging gives us the necessary condition:

While we were able to generalize our result to gradient descent for N = 2, our proof technique relies on the ability to get a non-implicit solution for ??(t) which we discretized and bounded.

This is harder to generalize to larger values of N , where the solution is implicit.

Still, we can informally illustrate the effect of depth on the dynamics of gradient descent by approximating the update rule of the values.

We start by reminding ourselves of the gradient descent update rule for ??, for a learning rate ?? = c

To compare two values in the same scales, we will divide both sides by the optimal value ?? * i and look at the update step of the ratio r i =

We will focus on the early stages of the optimization process, where r 1.

This means we can neglect the 1 ??? r i (t) term in the update step, giving us the approximate update step we will use to compare the general i, j values:

We would like to compare the dynamics of r i and r j , which is difficult to do when the recurrence relation isn't solvable.

However, we can observe the first iteration of gradient descent and see how depth affects this iteration.

Since we are dealing with variables which are ratios of different optimal values, the initial values of r are different.

Denoting r = ?? * i ?? * j , we can describe the initialization of r j using that of r i :

Plugging in the initial conditions and noting that R i = rR j , we get:

We see that the two ratios have a similar update, with the ratio of optimal values playing a role in how large the initial value is versus how large the added value is.

When we use a small learning rate, we have a very small c which means we can make a final approximation and neglect the higher order terms of c:

We can see that while the initial conditions favor r j , the size of the update for r i is larger by a factor of r N ???1 N when the initialization and learning rates are small.

This accumulates throughout the optimization, making r i eventually converge faster than r j .

The effect of depth here is clear -the deeper the model, the larger the relative step size of r i and the faster it converges relative to r j .

Learning our toy model, when it's incremental learning is taken to the limit, can be described as an iterative procedure where at every step an additional feature is introduced such that it's weight is non-zero and then the model is optimized over the current set of features.

This description is also relevant for the sparse approximation algorithm orthogonal matching pursuit (Pati et al., 1993) , where the next feature is greedily chosen to be the one which most improves the current model.

While the toy model and OMP are very different algorithms for learning sparse linear models, we will show empirically that they behave similarly.

This allows us to view incremental learning as a continuous-time extension of a greedy iterative algorithm.

To allow for negative weights in our experiments, we augment our toy model as in the toy model of Woodworth et al. (2019) .

Our model will have the same induced form as before:

However, we parameterize ?? using w + , w ??? ??? R d in the following way:

We can now treat this algorithm as a sparse approximation pursuit algorithm -given a dictionary D ??? R d??n and an example x ??? R d , we wish to find the sparsest ?? for which D?? ??? x by minimizing the 0 norm of ?? subject to ||D?? ??? x|| 2 2 = 0 3 .

Under this setting, we can compare OMP to our toy model by comparing the sets of features that the two algorithms choose for a given example and dictionary.

In figure 3 we run such a comparison.

Using a dictionary of 1000 atoms and an example of dimensionality 80 sampled from a random hidden vector of a given sparsity s, we run both algorithms and record the first s features chosen 4 .

Figure 3: Empirical comparison of the dynamics of the toy model to OMP.

The toy model has a depth of 5 and was initialized with a scale of 1e-4 and a learning rate of 3e-3.

We compare the fraction of agreement between the sets of first s features selected of the two algorithms for every given sparsity level s, averaged over 100 experiments (the shaded regions are empirical standard deviations).

For example, for sparsity level 3, we look at the sets of first 3 features selected by each algorithm and calculate the fraction of them that appear in both sets.

For every sparsity s, we plot the mean fraction of agreement between the sets of features chosen by OMP and the toy model over 100 experiments.

We see that the two algorithms choose very similar features at the beginning, suggesting that the deep model approximates the discrete behavior of OMP.

Only when the number of features increases do we see that the behavior of the two models begins to differ, caused by the fact that the toy model has a finite initialization scale and learning rate.

These experiments demonstrate the similarity between the incremental learning of deep models and the discrete behavior of greedy approximation algorithms such as OMP.

Adopting this view also allows us to put our finger on another strength of the dynamics of deep models -while greedy algorithms such as OMP require the analytical solution or approximation of every iterate, the dynamics of deep models are able to incrementally learn any differentiable function.

For example, looking back at the matrix sensing task and the classification models in section 4, we see that while there isn't an immediate and efficient extension of OMP for these settings, the dynamics of learning deep models extends naturally and exhibits the same incremental learning as OMP.

Theorem.

Minimizing the deep matrix sensing model described in (5) with gradient flow over the depth normalized squared loss (6), with Gaussian inputs and weights initialized as in (5) leads to the following dynamical equations for different values of N :

Where ?? i and ?? * i are the ith singular values of W and W * , respectively, corresponding to the same singular vectors.

Proof.

We will adapt the proof from Saxe et al. (2013) for multilayer linear networks.

The gradient flow equations for W n , n ??? [N ] are: (3), U diagonalizes all W n matrices at initialization such that D n = U W n U T = N ??? ?? 0 I. Making this change of variables for all W n , we get:

Rearranging, we get a set of decoupled differential equations for the singular values of W n :

Note that since these matrices are all diagonal at initialization, the above dynamics ensure that they remain diagonal throughout the optimization.

Denoting ?? n,i as the i'th singular value of W n and ?? i as the i'th singular value of W , we get the following differential equation:

Since we assume at initialization that ???n, m, i : ?? n,i (0) = ?? m,i (0) = N ??? ?? 0 , the above dynamics are the same for all singular values and we get ???n, m, i : ?? n,i (t) = ?? m,i (t) = N ?? i (t).

We may now use this to calculate the dynamics of the singular value of W , since they are the product the the singular values of all W n matrices.

Denoting ?? ???n,i = k =n ?? k,i and using the chain rule:

Our analytical results are only applicable for the population loss over Gaussian inputs.

These conditions are far from the ones used in practice and studied in Arora et al. (2019) , where the problem is over-determined and the weights are drawn from a Gaussian distribution with a small variance.

To show our conclusions regarding incremental learning extend qualitatively to more natural settings, we empirically examine the deep matrix sensing model in this natural setting for different depths and initialization scales as seen in figure 4.

Notice how incremental learning is exhibited even when the number of examples is much smaller than the number of parameters in the model.

While we can't rely on our theory for describing the exact dynamics of the optimization for these kinds of over-determined problems, the qualitative conclusions we get from it are still applicable.

Another interesting phenomena we should note is that once the dataset becomes very small (the second row of the figure) , we see all "currently active" singular values change at the beginning of every new phase (this is best seen in the bottom-right panel).

This suggests that since there is more than one optimal solution, once we increase the current rank of our model it may find a solution that has a different set of singular values and vectors and thus all singular values change at the beginning of a new learning phase.

This demonstrates the importance of incremental learning for obtaining The columns correspond to different parameterization depths, while the rows correspond to different dataset sizes.

In both cases the problem is over-determined, since the number of examples is smaller than the number of parameters.

Since the original matrix is rank-4, we can recognize an unsuccessful recovery when all five singular values are nonzero, as seen clearly for both depth-1 plots.

sparse solutions -once the initialization conditions and depth are such that the learning phases are distinct, gradient descent finds the optimal rank-i solution in every phase i. For these dynamics to successfully recover the optimal solution at every phase, the phases need to be far enough apart from each other to allow for the singular values and vectors to change before the next phase begins.

Theorem.

Minimizing the quadratic network described and initialized as in (7) with gradient flow over the variance loss defined in (2) with Gaussian inputs leads to the following dynamical equations:??

Where ?? i and ?? * i are the ith singular values of W and W * , respectively, corresponding to the same singular vectors.

Proof.

Our proof will follow similar lines as the analysis of the deep matrix sensing model.

Taking the expectation of the variance loss over Gaussian inputs for our model gives us:

Following the gradient flow dynamics over W leads to the following differential equation:

We can now calculate the gradient flow dynamics of W T W using the chain rule:

Now, under our initialization W T 0 W 0 = ?? 0 I, we get that W T W and W T * W * are simultaneously diagonalizable at initialization by some matrix U , such that the following is true for diagonal D and D * :

Multiplying equation (10) by U and U T gives us the following dynamics for the singular values of

These matrices are diagonal at initialization, and remain diagonal throughout the dynamics (the offdiagonal elements are static according to these equations).

We may now look at the dynamics of a single diagonal element, noticing it is equivalent to the depth-2 toy model:

It may seem that the variance loss is an unnatural loss function to analyze, since it isn't used in practice.

While this is true, we will show how the dynamics of this loss function are an approximation of the square loss dynamics.

We begin by describing the dynamics of both losses, showing how incremental learning can't take place for quadratic networks as defined over the squared loss.

Then, we show how adding a global bias to the quadratic network leads to similar dynamics for small initialization scales.

in the previous section, we derive the differential equations for the singular values of W T W under the variance loss:??

We will now derive similar equations for the squared loss.

The scaled squared loss in expectation over the Gaussian inputs is:

Figure 5: Quadratic model's evolution of top-5 singular values for a rank-4 labeling function.

The rows correspond to whether or not a global bias is introduced to the model.

The first two columns are for a large dataset (one optimal solution) and the last two columns are for a small dataset (overdetermined problem).

When a bias is introduced, it is initialized to it's optimal value at initialization.

Note how without the bias, the singular values are learned together and there is over-shooting of the optimal singular value caused by the coupling of the dynamics of the singular values.

For the small datasets, we see that the model with no bias reaches a solution with a larger rank.

Once a global bias is introduced, the dynamics become more incremental as in the analysis of the variance loss.

Note that in this case the solution obtained for the small dataset is the optimal low-rank solution.

the bias isn't optimal, and so incremental learning can still take place (assuming a small enough initialization).

Under these considerations, we say that the dynamics of the squared loss for a quadratic network with an added global bias resemble the idealized dynamics of the variance loss for a depth-2 linear model which we analyze formally in the paper.

In figure 5 we experimentally show how adding a bias to a quadratic network does lead to incremental learning similar to the depth-2 toy model.

In section 4.3 we viewed our toy model as a special case of the deep diagonal networks described in Gunasekar et al. (2018) , expected to be biased towards sparse solutions.

Figure 6 shows the dynamics of the largest values of ?? for different depths of the model.

We see that the same type of incremental learning we saw in earlier models exists here as well -the features are learned one by one in deeper models, resulting in a sparse solution.

The leftmost panel shows how the initialization scale plays a role here as well, with the solution being more sparse when the initialization is small.

We should note that these results do not defy the results of Gunasekar et al. (2018) (from which we would expect the initialization not to matter), since their results deal with the solution at t ??? ???.

The linear circular-convolutional network of Gunasekar et al. (2018) deals with one-dimensional convolutions with the same number of outputs as inputs, such that the mapping from one hidden layer to the next is parameterized by w n and defined to be: shaded regions denoting empirical standard deviations.

We see that depth-1 models reach results similar to the max-margin SVM solution as predicted by Gunasekar et al. (2018) , while deeper models are highly correlated with the sparse solution, with this correlation increasing when the initialization scale is small.

The other panels show the evolution of the absolute values of the top-5 weights of ?? for the smallest initialization scale.

Note that as we increase the depth, incremental learning is clearly presented.

The final layer is a fully connected layer parameterized by w N ??? R d , such that the final model can be written in the following way: This lemma connects the convolutional network to the diagonal network, and thus we should expect to see the same incremental learning of the values of the diagonal network exhibited by the Fourier coefficients of the convolutional network.

In figure 7 we see the same plots as in figure 6 but for the Fourier coefficients of the convolutional model.

We see that even when the model is far from the toy parameterization (there is no weight sharing and the initialization is with random Gaussian weights), incremental learning is still clearly seen in the dynamics of the model.

We see how the inherent reason for the sparsity towards sparse solution found in Gunasekar et al. (2018) is the result of the dynamics of the model -small amplitudes are attenuated while large ones are amplified.

Published as a conference paper at ICLR 2020 We see that depth-1 models reach results similar to the max-margin SVM solution, while deeper models are highly correlated with the optimal sparse solution.

The other panels show the evolution of the amplitudes of the top-5 frequencies of ?? for the smallest initialization scale.

Note that as we increase the depth, incremental learning is clearly presented.

@highlight

We study the sparsity-inducing bias of deep models, caused by their learning dynamics.