Abstract Stochastic gradient descent (SGD) and Adam are commonly used to optimize deep neural networks, but choosing one usually means making tradeoffs between speed, accuracy and stability.

Here we present an intuition for why the tradeoffs exist as well as a method for unifying the two in a continuous way.

This makes it possible to control the way models are trained in much greater detail.

We show that for default parameters, the new algorithm equals or outperforms SGD and Adam across a range of models for image classification tasks and outperforms SGD for language modeling tasks.

One of the most common methods of training neural networks is stochastic gradient descent (SGD) (Bottou et al. (2016) ).

SGD has strong theoretical guarantees, including convergence in locally non-convex optimization problems (Lee et al. (2016) ).

It also shows improved generalization and stability when compared to other optimization algorithms (Smith & Le (2018) ).

There have been various efforts in improving the speed and generalization of SGD.

One popular modification is to use an adaptive gradient (Duchi et al. (2011) ), which scales the gradient step size to be larger in directions with consistently small gradients.

Adam, an implementation that combines SGD with momentum and an adaptive step size inversely proportional to the RMS gradient, has been particularly successful at speeding up training and solving particular problems (Kingma & Ba (2014) ).

However, at other problems it pays a penalty in worse generalization (Wilson et al. (2017) ; Keskar & Socher (2017) ), and it requires additional modifications to achieve a convergence guarantee (Reddi et al. (2018) ; Li & Orabona (2018) ).

Here we develop an intuition for adaptive gradient methods that allows us to unify Adam with SGD in a natural way.

The new optimizer, SoftAdam, descends in a direction that mixes the SGD with Adam update steps.

As such, it should be able to achieve equal or better optimization results across a variety of problems.

Several authors have recently tried to combine Adam and SGD to get the best of both worlds.

However, these have not always enabled better generalization or performance across different problems.

In one study, the optimization algorithm was switched from Adam to SGD during training based on a scale-free criterion, preventing the addition of a new hyper-parameter (Keskar & Socher (2017) ).

The result is that the longer the convolutional networks were trained on Adam, the worse their generalization performance compared to SGD.

The best performance came from switching to SGD in this case as soon as possible.

Another recent algorithm takes the approach of clipping the large Adam updates to make them more similar to SGD as the training nears the end (Luo et al. (2019) ).

However, this approach requires two new hyper-parameters: the rate at which the training is switched over, and a learning rate for both SGD and Adam.

Similar to this work, partially adaptive methods (Chen & Gu (2018) ) can allow arbitrary mixing between SGD and Adam.

However, in that work the step size is not strictly smaller than the SGD step and so the same guarantees cannot be made about convergence.

It is of interest to see whether there is any advantage over these methods.

Other algorithms have shown improvements on SGD by averaging weights over many steps (Polyak & Juditsky (1992); Zhang et al. (2019); Izmailov et al. (2018) ).

These algorithms are complementary to the algorithm developed here, as they require an underlying algorithm to choose the step direction at any point.

The fundamental idea of gradient descent is to follow the path of steepest descent to an optimum.

Stochastic gradient descent enables us to optimize much larger problems by using randomly subsampled training data.

The stochastic gradient descent algorithm will minimize the loss function J(θ; x), which is parameterized by θ and takes as input training data x,

where α is a learning rate that may vary with t and x t is the training data selected for the batch at step t. The convergence rate can be improved further by using a running average of the gradient, initializing with m 0 ← 0.

This method, known as momentum (Goh (2017) ), may be written as,

A further development that has improved convergence, especially for LSTM and language modeling tasks, involves the second gradient as well.

This specific version is known as the Adam algorithm (Kingma & Ba (2014) ),

are unbiased estimators of the first and second moment respectively.

In order to analyze the convergence of these algorithms, we can consider a second-order approximation of J on its combined argument z = (θ; x) in the region of (θ t ; x t ),

where H t is the Hessian of J(z) around z t .

This gives us the gradient,

which becomes the SGD update step,

Unrolling this update step can be shown to lead to an expression for the distance from the optimal value z , in the basis of Hessian eigenvectors ξ i :

We can see that the learning is stable if the learning rate α satisfies,

In addition, we find that the value for the learning rate that leads to the fastest overall convergence is,

where λ 1 and λ n are the max and min eigenvalues of H, respectively.

If rather than a single learning rate α, we were to use a diagonal matrix D such that the update is,

we may be able to modify the diagonal entries

such that faster overall convergence is achieved.

For example, in the special case that the Hessian is diagonal, the convergence rate for the i-th element becomes,

i .

In this situation, if the eigenvalues λ i are known, the algorithm can converge to the minimum in exactly one step.

This corresponds with some intuition behind adaptive moment methods: that taking a step with a "constant" size in every direction toward the target will reach convergence faster than taking a step proportional to the gradient size.

Because the eigenvalues and eigenvectors not known a priori, for a practical algorithm we must rely on an approximation to find d i .

One technique named AdaGrad (Duchi et al. (2011) ) prescribes the diagonal elements:

For building our intuition, we consider the special case where the Hessian is diagonal,

Combining this with Equation 2, we compare the AdaGrad coefficient to the optimal value for

As long as αλ i , this will be true when,

This may be true on average if the initializations z 0i and optima z i can be made over the same subspaces.

That is, if z i is uncorrelated to λ i , we can expect this to have good performance on average.

However, there can be significant errors in both overestimating and underestimating the eigenvalue.

One would expect that for a typical problem b i and λ i might be drawn from uncorrelated distributions.

In this case, large values of z i will be likely to correspond to small values of λ i .

Since z 0 can only be drawn from the average distribution (no information is known at this point), the estimated λ i is more likely to be large, as the initialization will be far from the optimum.

Intuitively, the gradient is large because the optimum is far from the initialization, but the algorithm mistakes this large gradient for a large eigenvalue.

On the other hand, when the parameter is initialized close to its optimum, the algorithm will mistakenly believe the eigenvalue is small, and so take relatively large steps.

Although they do not affect the overall convergence much on their own (since the parameter is near its optimum), these steps can add significant noise to the process, making it difficult to accurately measure gradients and therefore find the optimum in other parameters.

This problem will be significantly worse for Adam, which forgets its initialization with some decay factor β 2 .

In that case, as each parameter reaches its optimum, its estimated eigenvalue λ i drops and the step size gets correspondingly increased.

In fact, the overall algorithm can be divergent as each parameter reaches its optimum, as the step size will grow unbounded unless α is scheduled to decline properly or a bounding method like AMSGrad is used (Reddi et al. (2018)).

In addition, reviewing our earlier assumption of small , these algorithms will perform worse for small eigenvalues λ i < /α.

This might be especially bad in Adam where late in training when the

We finally note that the Hessians in deep learning problems are not diagonal (Sagun et al. (2016); Li et al. (2019) ).

As such, each element might be better optimized by a learning rate that better serves both its min and max eigenvalues.

Overall, this understanding has led us to believe that adaptive moments might effectively estimate λ i when it is large, but might be less effective when it is small.

In order to incorporate this information about large eigenvalues, as well as optimize the learning rate to account for variation in the eigenvalues contributing to convergence of a particular component, we consider the an update to Eq. 1,

whereλ is an average eigenvalue and η is a new hyper-parameter that controls the weighting of the eigenvalue estimation.

Here we have addedλ to the numerator so that α does not need to absorb the changes to the RMS error as it does in Eq. 4.

This also recovers the SGD convergence guarantees, since the step is always within a factor of η to an SGD step.

In addition, this will allow us to recover SGD with momentum exactly in the limit η → 0.

We use the adaptive gradient estimation,

wherev t is the mean value of v t , to write,

One issue with the above estimation is that its variance is very large at the beginning of training (Liu et al. (2019) ).

It was suggested that this is the reason that warmup is needed for Adam and shown

Input: θ 0 ∈ F: initial parameters, {α t > 0} T t=1 : learning rate, α wd , β 1 , β 2 , η, : other hyperparameters, J t (θ): loss function Output:

Average over all elements

Calculate the denominator

Perform the update end for return θ T that rectifying it can make warmup unnecessary.

Where v t is the average of n t elements and v ∞ the average of n ∞ , we define r t = n t /n ∞ and:

This finally forms the basis for our algorithm.

Our algorithm differs from Adam in a few other ways.

First, the biased gradient estimate is used rather than the unbiased one.

This matches the SGD implementation of momentum, and also avoids magnifying the large variance of early gradient estimates:

In addition, the second moment v t is calculated in an unbiased way using an effectiveβ 2 (t), or by abuse of notationβ 2t :β

This has a negligable impact on the performance of the algorithm, but makes tracking the moment over time easier since it does not need to be un-biased later.

We then calculate the ratio of the number of samples n t used to calculate the moment v t to the steady state number of samples in the average n ∞ :

We finally note that the weight decay should be calculated separately from this update as in AdamW (Loshchilov & Hutter (2017) ).

In order to test the ability of this algorithm to reach better optima, we performed testing on a variety of different deep learning problems.

In these problems we keep η at the default value of 1.

Because the algorithm is the same as SGDM if η = 0 and is comparable to Adam when η = −1 , getting results at least as good as those algorithms is just a matter of parameter tuning.

These results are intended to show that SoftAdam performs remarkably well with a common parameter choice.

For the best performance, the hyper-parameter η and learning rate schedule α should be optimized for a particular problem.

We trained a variety of networks:

1 AlexNet (Krizhevsky et al. (2012) ), VGG19 with batch normalization (Simonyan & Zisserman (2014) ), ResNet-110 with bottleneck blocks (He et al. (2015) ), PreResNet-56 with bottleneck blocks (He et al. (2016) ), DenseNet-BC with L=100 and k=12 (Huang et al. (2016) ) on the CIFAR-10 dataset (Krizhevsky (2012) ) using SGD, AdamW and SoftAdam.

For each model and optimization method, the weight decay was varied over [1e-4,2e-4,5e-4,1e-3,2e-3,5e-3] .

For AdamW the learning rate was varied over [1e-4,2e-4,5e-4,1e-3,2e-3] .

For each optimizer and architecture and the best result was chosen, and three runs with separate initializations were used go generate the final data.

The learning schedule reduced the learning rate by a factor of 10 at 50% and 75% through the total number of epochs.

The results are summarized in Table 1.

We find that SoftAdam equals or outperforms SGD in training classifiers on this dataset.

Due to the larger optimal weight decay constant, SoftAdam achieves lower validation loss at a higher train loss than SGD.

We trained a 3-layer LSTM with 1150 hidden units per layer on the Penn Treebank dataset (Mikolov et al. (2010) ) in the same manner as Merity et al. (2017) .

For SoftAdam the weight drop was increased from 0.5 to 0.6.

Results for the average of three random initializations are shown in Figure 2 (a) and are summarized in Table 2 .

For these parameters, SoftAdam outperforms SGD significantly but does not quite achieve the same results as Adam.

Note that for this experiment we chose Adam instead of AdamW for comparison due to its superior performance.

We also trained a transformer using the fairseq package by Ott et al. (2019) on the IWSLT'14 German to English dataset.

Results for each method with optimized hyperparameters are summarized in Table 3 .

Note that no warmup is used for training SoftAdam, but warmup is used for AdamW and

In this paper, we have motivated and demonstrated a new optimization algorithm that naturally unifies SGD and Adam.

We have focused our empirical results on the default hyper-parameter setting, η = 1, and predetermined learning schedules.

With these parameters, the algorithm was shown to produce optimization that is better than or equal to SGD and Adam on image classification tasks.

It also performed significantly better than SGD on language modeling tasks.

Together with finding the optimal values for η, we expect a better understanding of the learning schedule to bring light to the way in which the adaptive gradient methods improve convergence.

SoftAdam now also makes it possible to create a learning schedule on η, which may be another fruitful avenue of research, expanding on the work of Ward et al. (2018) .

Better understanding of how adaptive gradients improve the convergence of practical machine learning models during training will enable larger models to be trained to more accurately in less time.

This paper provides a useful intuition for how that occurs and provides a new algorithm that can be used to improve performance across a diverse set of problems.

# S t a t e i n i t i a l i z a t i o n i f l e n ( s t a t e ) == 0 : s t a t e [ " s t e p " ] = 0 # E x p o n e n t i a l moving a v e r a g e o f g r a d i e n t v a l u e s s t a t e [ " e x p a v g " ] = t o r c h .

z e r o s l i k e ( p .

d a t a ) # E x p o n e n t i a l moving a v e r a g e o f # s q u a r e d g r a d i e n t v a l u e s s t a t e [ " e x p a v g s q " ] = t o r c h .

z e r o s l i k e ( p .

d a t a ) e x p a v g , e x p a v g s q = ( s t a t e [ " e x p a v g " ] , s t a t e [ " e x p a v g s q " ] , ) b e t a 1 , b e t a 2 = g r o u p [ " b e t a s " ] s t a t e [ " s t e p " ] += 1 b e t a 2 h a t = min ( b e t a 2 , 1 .

0 − 1 .

0 / ( s t a t e [ " s t e p " ] ) )

r b e t a = ( 1 − b e t a 2 ) / ( 1 − b e t a 2 h a t ) e t a h a t 2 = ( g r o u p [

" e t a " ] * g r o u p [ " e t a " ] * r b e t a ) # Decay t h e f i r s t and s e c o n d moment w i t h t h e # r u n n i n g a v e r a g e c o e f f i c i e n t e x p a v g .

mul ( b e t a 1 ) .

a d d r e t u r n l o s s

<|TLDR|>

@highlight

An algorithm for unifying SGD and Adam and empirical study of its performance