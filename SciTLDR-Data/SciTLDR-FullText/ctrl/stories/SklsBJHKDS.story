In this work, we aim to solve data-driven optimization problems, where the goal is to find an input that maximizes an unknown score function given access to a dataset of input, score pairs.

Inputs may lie on extremely thin manifolds in high-dimensional spaces, making the optimization prone to falling-off the manifold.

Further, evaluating the unknown function may be expensive, so the algorithm should be able to exploit static, offline data.

We propose model inversion networks (MINs) as an approach to solve such problems.

Unlike prior work, MINs scale to extremely high-dimensional input spaces and can efficiently leverage offline logged datasets for optimization in both contextual and non-contextual settings.

We show that MINs can also be extended to the active setting, commonly studied in prior work, via a simple, novel and effective scheme for active data collection.

Our experiments show that MINs act as powerful optimizers on a range of contextual/non-contextual, static/active problems including optimization over images and protein designs and learning from logged bandit feedback.

Data-driven optimization problems arise in a range of domains: from protein design (Brookes et al., 2019) to automated aircraft design (Hoburg & Abbeel, 2012) , from the design of robots (Liao et al., 2019) to the design of neural net architectures (Zoph & Le, 2017) and learning from logged feedback, such as optimizing user preferences in recommender systems.

Such problems require optimizing unknown reward or score functions using previously collected data consisting of pairs of inputs and corresponding score values, without direct access to the score function being optimized.

This can be especially challenging when valid inputs lie on a low-dimensional manifold in the space of all inputs, e.g., the space of valid aircraft designs or valid images.

Existing methods to solve such problems often use derivative-free optimization (Snoek et al.) .

Most of these techniques require active data collection where the unknown function is queried at new inputs.

However, when function evaluation involves a complex real-world process, such as testing a new aircraft design or evaluating a new protein, such active methods can be very expensive.

On the other hand, in many cases there is considerable prior data -existing aircraft and protein designs, and advertisements and user click rates, etc.

-that could be leveraged to solve the optimization problem.

In this work, our goal is to develop an optimization approach to solve such optimization problems that can (1) readily operate on high-dimensional inputs comprising a narrow, low-dimensional manifold, such as natural images, (2) readily utilize offline static data, and (3) learn with minimal active data collection if needed.

We can define this problem setting formally as the optimization problem

where the function f (x) is unknown, and we have access to a dataset D = {(x 1 , y 1 ), . . . , (x N , y N )}, where y i denotes the value f (x i ).

If no further data collection is possible, we call this the data-driven model-based optimization setting.

This can also be extended to the contextual setting, where the aim is to optimize the expected score function value across a context distribution.

That is,

where π maps contexts c to inputs x, such that the expected score under the context distribution p 0 (c) is optimized.

As before, f (c, x) is unknown and we have access to a dataset D = {(c i ,

, where y i is the value of f (c i , x i ).

Such contextual problems with logged datasets have been studied in the context of contextual bandits Joachims et al., 2018) .

A simple way to approach these model-based optimization problems is to train a proxy function f θ (x) or f θ (c, x), with parameters θ, to approximate the true score, using the dataset D. However, directly using f θ (x) in place of the true function f (x) in Equation (1) generally works poorly, because the optimizer will quickly find an input x for which f θ (x) outputs an erroneously large value.

This issue is especially severe when the inputs x lie on a narrow manifold in a high-dimensional space, such as the set of natural images (Zhu et al., 2016) .

The function f θ (x) is only valid near the training distribution, and can output erroneously large values when queried at points chosen by the optimizer.

Prior work has sought to addresses this issue by using uncertainty estimation and Bayesian models (Snoek et al., 2015) for f θ (x), as well as active data collection (Snoek et al.) .

However, explicit uncertainty estimation is difficult when the function f θ (x) is very complex or when x is high-dimensional.

Instead of learning f θ (x), we propose to learn the inverse function, mapping from values y to corresponding inputs x. This inverse mapping is one-to-many, and therefore requires a stochastic mapping, which we can express as f −1 θ (y, z) → x, where z is a random variable.

We term such models model inversion networks (MINs).

MINs provide us with a number of desirable properties: they can utilize static datasets, handle high-dimensional input spaces such as images, can handle contextual problems, and can accommodate both static datasets and active data collection.

We discuss how to design simple active data collection methods for MINs, leverage advances in deep generative modeling (Goodfellow et al.; Brock et al., 2019) , and scale to very high-dimensional input spaces.

We experimentally demonstrate MINs in a range of settings, showing that they outperform prior methods on high-dimensional input spaces, perform competitively to Bayesian optimization methods on tasks with active data collection and lower-dimensional inputs, and substantially outperform prior methods on contextual optimization from logged data (Swaminathan & Joachims, a) .

Bayesian optimization.

In this paper, we aim to solve data-driven optimization problems.

Most prior work aimed at solving such optimization problems has focused on the active setting.

This includes algorithms such as the cross entropy method (CEM) and related derivative-free methods Rubinstein (1996) ; Rubinstein & Kroese (2004) , reward weighted regression Peters & Schaal, Bayesian optimization methods based on Gaussian processes Shahriari et al. (2016) ; , and variants that replace GPs with parametric acquisition function approximators such as Bayesian neural networks (Snoek et al., 2015) and latent variable models (Kim et al., 2019; Garnelo et al., 2018b; a) , as well as more recent methods such as CbAS (Brookes et al., 2019) .

These methods require the ability to query the true function f (x) at each iteration to iteratively arrive at a near-optimal solution.

We show in Section 3.3 that MINs can be applied to such an active setting as well, and in our experiments we show that MINs can perform competitively with these prior methods.

Additionally, we show that MINs can be applied to the static setting, where these prior methods are not applicable.

Furthermore, most conventional BO methods do not scale favourably to high-dimensional input spaces, such as images, while MINs can handle image inputs effectively.

Contextual bandits.

Equation 2 captures the class of contextual bandit problems.

Prior work on batch contextual bandits has focused on batch learning from bandit feedback (BLBF), where the learner needs to produce the best possible policy that optimizes the score function from logged experience.

Existing approaches build on the counterfactual risk minimization (CRM) principle (Swaminathan & Joachims, a;b) , and have been extended to work with deep nets (Joachims et al., 2018) .

In our comparisons, we find that MINs substantially outperform these prior methods in the batch contextual bandit setting.

Deep generative modeling.

Recently, deep generative modeling approaches have been very successful at modelling high-dimensional manifolds such as natural images (Goodfellow et al.; Van Den Oord et al.; Dinh et al., 2016) , speech (van den Oord et al., 2018), text (Yu et al.) , alloy composition prediction (Nguyen et al.) , etc.

MINs combine the strength of such generative models with important algorithmic decisions to solve model-based optimization problems.

In our experimental evaluation, we show that these design decisions are important for adapting deep generative models to model-based optimization, and it is difficult to perform effective optimization without them.

In this section, we describe our model inversion networks (MINs) method, which can perform both active and passive model-based optimization over high-dimensional input spaces.

Problem statement.

Our goal is to solve optimization problems of the form x = arg max x f (x), where the function f (x) is not known, but we must instead use a dataset of input-output tuples D = {(x i , y i )}.

In the contextual setting described in Equation (2), each datapoint is also associated with a context c i .

For clarity, we present our method in the non-contextual setting, but the contextual setting can be derived analogously by conditioning all functions on the context.

In the active setting, which is most often studied in prior work, the algorithm is allowed to actively query f (x) one or more times on each iteration to augment the dataset, while in the static setting, only an initial static dataset is available.

The goal is to obtain the best possible x (i.e., the one with highest possible value of f (x )).

One naïve way of solving MBO problems is to learn a proxy score function f θ (x), via standard empirical risk minimization.

We could then maximize this learned function with respect to x via standard optimization methods.

However, naïve applications of such a method would fail for two reasons.

First, the proxy function f θ (x) may not be accurate outside the samples on which it is trained, and optimization with respect to it may simply lead to values of x for which f θ (x) makes the largest mistake in the negative direction.

The second problem is more subtle.

When x lies on a narrow manifold in very high-dimensional space (such as the space of natural images), the optimizer can produce invalid values of x, which result in arbitrary outputs when fed into f θ (x).

Since the shape of this manifold is unknown, it is difficult to constrain the optimizer to prevent this.

This second problem is rarely addressed or discussed in prior work, which typically focuses on optimization over low-dimensional and compact domains with known bounds.

Part of the reason for the brittleness of the naïve approach above is that f θ (x) has a high-dimensional input space, making it easy for the optimizer to find inputs x for which the proxy function produces an unreasonable output.

Can we instead learn a function with a small input space, which implicitly understands the space of valid, in-distribution values for x?

The main idea behind our approach is to model an inverse map that produces a value of x given a score value y, given by f −1 θ : Y → X .

The input to the inverse map is a scalar, making it comparatively easy to constrain to valid values, and by directly generating the inputs x, an approximation to the inverse function must implicitly understand which input values are valid.

As multiple x values can correspond to the same y, we design f −1 θ as a stochastic map that maps a score value along with a d z -dimensional random vector to a x, f −1 θ : Y × Z → X , where z is distributed according to a prior distribution p 0 (z).

To define the inverse map objective, let the data distribution be denoted p D (x, y), let p D (y) be the marginal over y, and let p(y) be an any distribution defined on Y (which could be equal to p D (y)).

We can train the proxy inverse map f −1 θ under distribution p(y) by minimizing the following objective:

where p f

, and D is a measure of divergence between the two distributions.

Using the Kullback-Leibler divergence leads to maximum likelihood learning, while Jensen-Shannon divergence motivates a GAN-style training objective.

MINs can be adapted to the contextual setting by passing in the context as an input and learning f −1 θ (y i , z, c i ).

In standard empirical risk minimization, we would choose p(y) to be the data distribution p D (y), such that the expectation be approximated simply by sampling training tuples (x i , y i ) from the training set.

However, as we will discuss in Section 3.3, a more careful choice for p(y) can lead to better performance.

The MIN algorithm is based on training an inverse map, and then using it via the inference procedure in Section 3.2 to infer the x that approximately optimizes f (x).

The structure of the MIN algorithm is shown in Algorithm 1.

Once the inverse map is trained, the goal of our algorithm is to generate the best possible x , which will maximize the true score function as well as possible under the dataset.

Since a score y needs to be provided as input to the inverse map, we must select for which score y to query the inverse map to obtain a near-optimal x. One naïve heuristic is to pick the best y max ∈ D and produce x max ∼ f −1 θ (y * max ) as the output.

However, the method should be able to extrapolate beyond the best score seen in the dataset, especially in contextual settings, where a good score may not have been observed for all contexts.

In order to extrapolate as far as possible, while still staying on the valid data manifold, we need to measure the validity of the generated values of x. One way to do this is to measure the agreement between the learned inverse map and an independently trained forward model f θ : the values of y for which the generated samples x are predicted to have a score similar to y are likely in-distribution, whereas those where the forward model predicts a very different score may be too far outside the training distribution.

Since the latent variable z captures the multiple possible outputs of the one-tomany inverse map, we can further optimize over z for a given y to find the best, most trustworthy output x. This can be formalized as the following optimization: y * ,z * := arg max

This optimization can be motivated as finding an extrapolated score that corresponds to values of x that lie on the valid input manifold, and for which independently trained forward and inverse maps agree.

Although this optimization uses an approximate forward map f θ (x), we show in our experiments in Section 4 that it produces substantially better results than optimizing with respect to a forward model alone.

The inverse map substantially constraints the search space, requiring an optimization over a 1-dimensional y and a (relatively) low-dimensional z, rather than the full space of inputs.

This scheme can be viewed as a special (deterministic) case of a probabilistic optimization procedure described in Appendix A.

A naïve implementation of the training objective in Equation (3) samples y from the data distribution p D (y).

However, as we are most interested in the inverse map's predictions for high values of y, it is much less important for the inverse map to predict accurate x values for values of y that are far from the optimum.

We could consider increasing the weights on datapoints with larger values of y.

In the extreme case, we could train only on the best datapoint -either the single datapoint with the largest y or, in the contextual case, the datapoint with the largest y for each context.

More generally, we can define the optimal y distribution p * (y), which is simply the delta function centered on the best y, p * (y) = δ y * (y), in the deterministic case.

If we instead assume that the observed scores have additive noise (i.e., we observe f (x) + ε, ε ∼ N ), then p * (y) would be a distribution centered around the optimal y. Of course, training on p * (y) is not practical, since it heavily down-weights most of the training data, leading to a very high-variance training objective, and is not even known in general, since the optimal data point is likely not in our training set.

In this section, we will propose a better choice for p(y) that trades off the variance due to an overly peaked training distribution and the bias due to training on the "wrong" distribution (i.e., anything other than p * (y)).

We can train under a distribution other than the empirical distribution by using importance sampling, such that we sample from p D and assign an importance weight, given by

.

By bounding the variance and the bias of the gradient of L p (D) estimate, with respect to the reweighted objective without sampling error under y drawn from p * (y), we obtain the following result: (Proof in Appendix B) Theorem 3.1 ((Informal) Bias + variance bound in MINs).

Let L(p * ) be the objective under p * (y) without sampling error:

Let N y be the number of datapoints with the particular y value observed in D, For some constants C 1 , C 2 , C 3 , with high confidence,

Theorem 3.1 suggests a tradeoff between being close to the optimal distribution p * (y) and reducing variance by covering the full data distribution p D .

We observe that the distribution p(y) that minimizes the RHS bound in Theorem 3.1 has the following form:

linear function of p * (y) that ensures that the distributions p and p * are close.

Theoretically, g(•) is an increasing, piece-wise linear function of •. We can interpret the expression for p(y) as a product of two likelihoods -the optimality of a particular y value and the likelihood of a particular y not being rare in D. We empirically choose an exponential parameteric form for this function, which we describe in Section 3.5.

This upweights the samples with higher scores, reduces the weight on rare y-values (i.e., those with low N y ), while preventing the weight on common y-values from growing, since Ny Ny+K saturates to 1 for large N y .

This is consistent with our intuition: we would like to upweight datapoints with high y-values, provided the number of samples at those values is not too low.

Of course, for continuous-valued scores, we rarely see the same score twice.

Therefore, we bin the y-values into discrete bins for the purpose of weighting, as we discuss in Section 3.5.

While the passive setting requires care in finding the best value of y for the inverse map, the active setting presents a different challenge: choosing a new query point x at each iteration to augment the dataset D and make it possible to find the best possible optimum.

Prior work on bandits and Bayesian optimization often uses Thompson sampling (TS) (Russo & Van Roy, 2016; Russo et al., 2018; Srinivas et al.) as the data-collection strategy.

TS maintains a posterior distribution over functions p(f t |D 1:t ).

At each iteration, it samples a function from this distribution and queries the point x t that greedily minimizes this function.

TS offers an appealing query mechanism, since it achieves sub-linear Bayesian regret (defined as the expected cumulative difference between the value of the optimal input and the selected input), given by O(

where T is the number of queries.

Maintaining a posterior over high-dimensional parametric functions is generally intractable.

However, we can devise a scheme to approximate Thompson sampling with MINs.

To derive this method, first note that sampling f t from the posterior is equivalent to sampling (x, y) pairs consistent with f tgiven sufficiently many (x, y) pairs, there is a unique smooth function f t that satisfies

For example, we can infer a quadratic function exactly from three points.

For a more formal description, we refer readers to the notion of Eluder dimension (Russo & Van Roy) .

Thus, instead of maintaining intractable beliefs over the function, we identify a function by the samples it generates, and define a way to sample synthetic (x, y) points such that they implicitly define a unique function sample from the posterior.

To apply this idea to MINs, we train the inverse map f −1 θt at each iteration t with an augmented

is a dataset of synthetically generated input-score pairs corresponding to unseen y values in D t .

Training f −1 θt on D t corresponds to training f −1 θt to be an approximate inverse map for a function f t sampled from p(f t |D 1:t ), as the synthetically generated samples S t implicitly induce a model of f t .

We can then approximate Thompson sampling by obtaining x t from f −1 θt , labeling it via the true function, and adding it to D t to produce D t+1 .

Pseudocode for this method, which we call "randomized labeling," is presented in Algorithm 2.

In Appendix C, we further derive O( √ T ) regret guarantees under mild assumptions.

Implementationwise, this method is simple, does not require estimating explicit uncertainty, and works with arbitrary function classes, including deep neural networks.

corresponding to unseen data points yi (by randomly pairing noisy observed xi values with unobserved y values.)

4:

Train inverse map f −1 t on D t = Dt ∪ St, using reweighting described in Section 3.3.

5:

Query function f at xt = f

Observe outcome: (xt, f (xt)) and update Dt+1 = Dt ∪ (xt, f (xt)) 7: end for

In this section, we describe our instantiation of MINs for high-dimensional inputs with deep neural network models.

GANs (Goodfellow et al.) have been successfully used to model the manifold of high-dimensional inputs, without the need for explicit density modelling and are known to produce more realistic samples than other models such as VAEs (Kingma & Welling, 2013) or Flows (Dinh et al., 2016) .

The inverse map in MINs needs to model the manifold of valid x thus making GANs a suitable choice.

We can instantiate our inverse map with a GAN by choosing D in Equation 3 to be the Jensen-Shannon divergence measure.

Since we generate x conditioned on y, the discriminator is parameterized as Disc(x|y), and trained to output 1 for a valid (x, y) pair (i.e., where y = f (x) and x comes from the data) and 0 otherwise.

Thus, we optimize the following objective:

This model is similar to a conditional GAN (cGAN), which has been used in the context of modeling distribution of x conditioned on a discrete-valued label (Mirza & Osindero, 2014) .

As discussed in Section 3.3, we additionally reweight the data distribution using importance sampling.

To that end, we discretize the space Y into B discrete bins b 1 , · · · , b B and, following Section 3.3, weight each bin

, where N bi is the number of datapoints in the bin, y * is the maximum score observed, and τ is a hyperparameter. (After discretization, using notation from Section 3.3, for any y that lies in bin b, p

In the active setting, we perform active data collection using the synthetic relabelling algorithm described in Section 3.4.

In practice, we train two copies of f −1 θ .

The first, which we call the exploration model f −1 expl , is trained with data augmented via synthetically generated samples (i.e., D t ).

The other copy, called the exploitation model f −1 exploit , is trained on only real samples (i.e., D t ).

This improves stability during training, while still performing data collection as dictated by Algorithm 2.

To generate the augmented dataset D t in practice, we sample y values from p * (y) (the distribution over high-scoring ys observed in D t ), and add positive-valued noise, thus making the augmented y values higher than those in the dataset which promotes exploration.

The corresponding inputs x are simply sampled from the dataset D t or uniformly sampled from the bounded input domain when provided in the problem statement. (for example, benchmark function optimization) After training, we infer best possible x from the trained model using the inference procedure described in Section 3.2.

In the active setting, the inference procedure is applied on f −1 exploit , the inverse map that is trained only on real data points.

The goal of our empirical evaluation is to answer the following questions.

(1) Can MINs successfully solve optimization problems of the form shown in Equations 1 and 2, in static settings and active settings, better than or comparable to prior methods?

(2) Can MINs generalize to high dimensional spaces, where valid inputs x lie on a lower-dimensional manifold, such as the space of natural images?

(3) Is reweighting the data distribution important for effective data-driven model-based optimization?

(4) Does our proposed inference procedure effectively discover valid inputs x with better values than any value seen in the dataset?

(5) Does randomized labeling help in active data collection?

We first study the data-driven model-based optimization setting.

This requires generating points that achieve a better function value than any point in the training set or, in the contextual setting, better than the policy that generated the dataset for each context.

We evaluate our method on a batch contextual bandit task proposed in prior work (Joachims et al., 2018) and on a high-dimensional contextual image optimization task.

We also evaluate our method on several non-contextual tasks that require optimizing over high-dimensional image inputs to evaluate a semantic score function, including hand-written characters and real-world photographs.

Batch contextual bandits.

We first study the contextual optimization problem described in Equation 2.

The goal is to learn a policy, purely from static data, that predicts the correct bandit arm x for each context c, such that the policy achieves a high overall score f (c, π(c)) on average across contexts drawn from a distribution p 0 (c).

We follow the protocol set out by Joachims et al. (2018) Joachims et al. (2018) , while the BanditNet column is our implementation; we were unable to replicate the performance from prior work (details in Appendix D).

MINs outperform both BanditNet and BanditNet * , both with and without the inference procedure in Section 3.2.

MINs w/o reweighting perform at par with full MINs on MNIST, and slightly worse on CIFAR 10, while still outperforming the baseline.

which evaluates contextual bandit policies trained on a static dataset for a simulated classification tasks.

The data is constructed by selecting images from the (MNIST/CIFAR) dataset as the context c, a random label as the input x, and a binary indicator indicating whether or not the label is correct as the score y. Multiple schemes can be used for selecting random labels for generating the dataset, and we evaluate on two such schemes, as described below.

We report the average score on a set of new contexts, which is equal to the average 0-1 accuracy of the learned model on a held out test set of images (contexts).

We compare our method to previously proposed techniques, including the BanditNet model proposed by Joachims et al. (2018) on the MNIST and CIFAR-10 (Krizhevsky, 2009) datasets.

Note that this task is different from regular classification, in that the observed feedback ((c i , x i , y i ) pairs) is partial, i.e. we do not observe the correct label for each context (image) c i , but only whether or not the label in the training tuple is correct or not.

We evaluate on two datasets:

(1) data generated by selecting random labels x i for each context c i and (2) data where the correct label is used 49% of the time, which matches the protocol in prior work (Joachims et al., 2018) .

We compare to BanditNet (Joachims et al., 2018) on identical dataset splits.

We report the average 0-1 test accuracy for all methods in Table 1 .

The results show that MINs drastically outperform BanditNet on both MNIST and CIFAR-10 datasets, indicating that MINs can successfully perform contextual model-based optimization in the static (data-driven) setting.

The results also show that utilizing the inference procedure in Section 3.2 produces an improvement of about 1.5% and 1.0% in test-accuracy on MNIST and CIFAR-10, respectively.

Character stroke width optimization.

In the next experiment, we study how well MINs optimize over high-dimensional inputs, where valid inputs lie on a lower-dimensional manifold.

We constructed an image optimization task out of the MNIST (LeCun & Cortes, 2010) dataset.

The goal is to optimize directly over the image pixels, to produce images with the thickest stroke width, such that the image corresponds either (a) to any valid character or (b) a valid instance of a particular character class.

A Figure 2: MIN optimization to obtain the youngest faces when trained on faces older than 15 (left) and older than 25 (right).

Generated faces (bottom) are obtained via inference in the inverse map at different points during model training.

Real faces of varying ages (including ages lower than those used to train the model) are shown in the top rows.

We overlay the actual age (negative of the score function) for each face on the real images, and the age obtained from subjective user rankings on the generated faces.

The score function being optimized (maximized) in this case is the negative age of the face.

successful algorithm will produce the thickest character that is still recognizable.

In Figure 1 , we observe that MINs generate images x that maximize the respective score functions in each case.

We also evaluate on a harder task where the goal is to maximize the number of disconnected blobs of black pixels in an image of a digit.

For comparison, we evaluate a method that directly optimizes the image pixels with respect to a forward model, of the form f θ (x).

In this case, the solutions are far off the manifold of valid characters.

We also compare to MINs without the reweighting scheme and the inference procedure, where y is the maximum possible y in the dataset to demonstrate the benefits of these two aspects.

Semantic image optimization.

The goal in these tasks is to quantify the ability of MINs to optimize high-level properties that require semantic understanding of images.

We consider MBO tasks on the IMDB-Wiki faces (Rothe et al., 2015; dataset, where the function f (x) is the negative of the age of the person in the image.

Hence, images with younger people have higher scores.

We construct two versions of this task: one where the training data consists of all faces older than 15 years, and the other where the model is trained on all faces older than 25 years.

This ensures that our model cannot simply copy the youngest face.

To obtain ground truth scores for the generated faces, we use subjective judgement from human participants.

We perform a study with 13 users.

Each user was asked to answer a set of 35 binary-choice questions each asking the user to pick the older image of the two provided alternatives.

We then fit an age function to this set of binary preferences, analogously to Christiano et al. (2017) .

Figure 2 shows the images produced by MINs.

For comparison, we also present some sample of images from the dataset partitioned by the ground truth score.

We find that the most likely age for optimal images produced by training MINs on images of people 15 years or older was 13.6 years, with the best image having an age of 12.2.

The model trained on ages 25 and above produced more mixed results, with an average age of 26.2, and a minimum age of 23.9.

We report these results in Table 2 .

This task is exceptionally difficult, since the model must extrapolate outside of the ages seen in the training set, picking up on patterns in the images that can be used to produce faces that appear younger than any face that the model had seen, while avoiding unrealistic images.

We also conducted experiments on contextual image optimization with MINs.

We studied contextual optimization over hand-written digits to maximize stroke width, using either the character category as the context c, or the top one-fourth or top half of the image.

In the latter case, MINs must learn to complete the image while maximizing for the stroke width.

In the case of class-conditioned optimization, MINs attain an average score over the classes of 237.6, while the dataset average is 149.0.

In the case where the context is the top half or quarter of the image, MINs obtain average scores of 223.57 and 234.32, respectively, while the dataset average is 149.0 for both tasks.

We report these results in Table 3 .

We also conducted a contextual optimization experiment on faces from the Celeb-A dataset, with some example images shown in Figure 3 .

The context corresponds to the choice for the attributes brown hair, black hair, bangs, or moustache.

The optimization score is given by the sum of the attributes wavy hair, eyeglasses, smiling, and no beard.

Qualitatively, we can see that MINs successfully optimize the score while obeying the target context, though evaluating the true score is impossible without subjective judgement on this task.

We discuss these experiments in more detail in Appendix D.1.

Figure 3 : Optimized x produced from contextual training on Celeb-A. Context = (brown hair, black hair, bangs, moustache and f (x) = 1(wavy hair, eyeglasses, smiling, no beard).

We show the produced x for two contexts.

The model optimizes score for both observed contexts such as brown or black hair and extrapolates to unobserved contexts such as brown and black hair.

In the active MBO setting, MINs must select which new datapoints to query to improve their estimate of the optimal input.

In this setting, we compare to prior model-based optimization methods, and evaluate the exploration technique described in Section 3.4.

Global optimization on benchmark functions.

We first compare MINs to prior work in Bayesian optimization on standard benchmark problems (DNGO) (Snoek et al., 2015) : the 2D Branin function, and the 6D Hartmann function.

As shown in Table 4 , MINs reach within ±0.1 units of the global minimum (minimization is performed here, instead of maximization), performing comparably with commonly used Bayesian optimization methods based on Gaussian processes.

We do not expect MINs to be as efficient as GP-based methods, since MINs rely on training parametric neural networks with many parameters, which is less efficient than GPs on low-dimensional tasks.

Exact Gaussian processes and adaptive Bayesian linear regression (Snoek et al., 2015) outperform MINs in terms of optimization precision and the number of samples queried, but MINs achieve comparable performance with about 4× more samples.

We also report the performance of MINs without the random labeling exploration method, instead selecting the next query point by greedily maximizing the current model with some additive noise.

We find that the random relabeling method produces substantially better results than the greedy data collection approach, indicating the importance of effective exploration methods for MINs.

Function Spearmint DNGO MIN MIN + greedy Branin (0.398) 0.398 ± 0.0 0.398 ± 0.0 0.398 ± 0.02 0.4 ± 0.05(800) Hartmann6 (-3.322) −3.3166 ± 0.02 −3.319 ± 0.00 −3.315 ± 0.05(600) −3.092 ± 0.12(1200) Protein fluorescence maximization.

In the next experiment, we study a high-dimensional active MBO task, previously studied by Brookes et al. (2019) .

This task requires optimizing over protein designs by selecting variable length sequences of codons, where each codon can take on one of 20 values.

In order to model discrete values, we use a Gumbel-softmax GAN also previously employed in (Gupta & Zou, 2018) , and as a baseline in (Brookes et al., 2019) .

For backpropagation, we choose a temperature τ = 0.75 for the Gumbel-softmax operation.

This is also mentioned in Appendix D. The aim in this task is to produce a protein with maximum fluorescence.

Each algorithm is provided with a starting dataset, and then allowed a identical, limited number of score function queries.

For each query made by an algorithm, it receives a score value from an oracle.

We use the trained oracles released by Brookes et al. (2019) .

These oracles are separately trained forward models, and can potentially be inaccurate, especially for datapoints not observed in the starting static dataset.

We compare to CbAS (Brookes et al., 2019) and other baselines, including CEM (Cross Entropy Method), RWR (Reward Weighted Regression) and a method that uses a forward model -GB (Gómez-Bombarelli et al., 2018) reported by Brookes et al. (2019) .

For evaluation, we report the groundtruth score of the output of optimization (max), and the 50th-percentile groundtruth score of all the samples produced via sampling (this is without inference in the MIN case) so as to be comparable to Brookes et al. (2019) .

In Table 5 , we show that MINs are comparable to the best performing method on this task, and produce samples with the highest score among all the methods considered.

These results suggest that MINs can perform competitively with previously proposed model-based optimization methods in the active setting, reaching comparable or better performance when compared both to Bayesian optimization methods and previously proposed methods for a higher-dimensional protein design task.

In this work, we presented a novel approach towards model-based optimization (MBO).

Instead of learning a proxy forward function f θ (x) from inputs x to scores y, MINs learn a stochastic inverse mapping from scores y to inputs.

MINs are resistent to out-of-distribution inputs and can optimize over high dimensional x values where valid inputs lie on a narrow manifold.

By using simple and principled design decisions, such as re-weighting the data distribution, MINs can perform effective model-based optimization even from static, previously collected datasets in the data-driven setting without the need for active data collection.

We also described ways to perform active data collection if needed.

Our experiments showed that MINs are capable of solving MBO optimization tasks in both contextual and non-contextual settings, and are effective over highly semantic score functions such as age of the person in an image.

Prior work has usually considered MBO in the active or "onpolicy" setting, where the algorithm actively queries data as it learns.

In this work, we introduced the data-driven MBO problem statement and devised a method to perform optimization in such scenarios.

This is important in settings where data collection is expensive and where abundant datasets exist, for example, protein design, aircraft design and drug design.

Further, MINs define a family of algorithms that show promising results on MBO problems on extremely large input spaces.

While MINs scale to high-dimensional tasks such as model-based optimization over images, and are performant in both contextual and non-contextual settings, we believe there are a number of interesting open questions for future work.

The interaction between active data collection and reweighting should be investigated in more detail, and poses interesting consequences for MBO, bandits and reinforcement learning.

Better and more principled inference procedures are also a direction for future work.

Another avenue is to study various choices of training objectives in MIN optimization.

In this section, we show that the inference scheme described in Equation 4, Section 3.2 emerges as a deterministic relaxation of the probabilistic inference scheme described below.

We re-iterate that in Section 3.2, a singleton x * is the output of optimization, however the procedure can be motivated from the perspective of the following probabilistic inference scheme.

Let p(x|y) denote a stochastic inverse map, and let p f (y|x) be a probabilistic forward map.

Consider the following optimization problem: arg max

where p θ (x|y) is the probability distribution induced by the learned inverse map (in our case, this corresponds to the distribution of f −1 θ (y, z) induced due to randomness in z ∼ p 0 (·)), p f (x|y) is the learned forward map, H is Shannon entropy, and D is KL-divergence measure between two distributions.

In Equation 4, maximization is carried out over the input y to the inverse-map, and the input z which is captured inp in the above optimization problem, i.e. maximization over z in Equation 4 is equivalent to choosingp subject to the choice of singleton/ Dirac-deltap.

The Lagrangian is given by:

In order to derive Equation 4, we restrictp to the Dirac-delta distribution generated by querying the learned inverse map f −1 θ at a specific value of z. Now note that the first term in the Lagrangian corresponds to maximizing the "reconstructed"ŷ similarly to the first term in Equation 4.

If p f is assumed to be a Gaussian random variable with a fixed variance, then log p f (ŷ|x) = −||ŷ − µ(x)|| Finally, in order to obtain the log p 0 (z) term, note that, D(p(x|y), p θ (x|y)) ≤ D(δ z (·), p 0 (·)) = − log p 0 (z) (by the data processing inequality for KL-divergence).

Hence, constraining log p 0 (z) instead of the true divergence gives us a lower bound on L. Maximizing this lower bound (which is the same as Equation 4) hence also maximizes the true Lagrangian L.

In this section, we provide details on the bias-variance tradeoff that arises in MIN training.

Our analysis is primarily based on analysing the bias and variance in the 2 norm of the gradient in two cases -if we had access to infinte samples of the distribution over optimal ys, p * (y) (this is a Dirac-delta distribution when function f (x) evaluations are deterministic, and a distribution with non-zero variance when the function evaluations are stochastic or are corrupted by noise).

Let

−1 (y j )) denote the empirical objective that the inverse map is trained with.

We first analyze the variance of the gradient estimator in Lemma B.2.

In order to analyse this, we will need the expression for variance of the importance sampling estimator, which is captured in the following Lemma.

Lemma B.1 (Variance of IS (Metelli et al., 2018) ).

Let P and Q be two probability measures on the space (X , F) such that d 2 (P ||Q) < ∞. Let x 1 , · · · , x N be N randomly drawn samples from Q, and f : X → R is a uniformly-bounded function.

Then for any δ ∈ (0, 1], with probability atleast 1 − δ,

Equipped with Lemma B.1, we are ready to show the variance in the gradient due to reweighting to a distribution for which only a few datapoints are observed.

θ .

Let N y denote the number of datapoints observed in D with score equal to y, and letL p (D) be as defined

, where the expectation is computed with respect to the dataset D.

Then, there exist some constants C 1 , C 2 such that with a confidence at least 1 − δ,

Proof.

We first bound the range in which the random variable ∇ θLp (D) can take values as a function of number of samples observed for each y. All the steps follow with high probability, i.e. with probability greater than 1 − δ,

is the exponentiated Renyi-divergence between the two distributions p and q, i.e.

dy.

The first step follows by applying Hoeffding's inequality on each inner term in the sum corresponding to y j and then bounding the variance due to importance sampling ys finally using concentration bounds on variance of importance sampling using Lemma B.1.

Thus, the gradient can fluctuate in the entire range of values as defined above with high probability.

Thus, with high probability, atleast 1 − δ,

The next step is to bound the bias in the gradient that arises due to training on a different distribution than the distribution of optimal ys, p * (y).

This can be written as follows:

where D TV is the total variation divergence between two distributions p and p * , and L is a constant that depends on the maximum magnitude of the divergence measure D. Combining Lemma B.2 and the above result, we prove Theorem 3.1.

In this section, we explain in more detail the randomized labeling algorithm described in Section 3.4.

We first revisit Thompson sampling, then provide arguments for how our randomized labeling algorithm relates to it, highlight the differences, and then prove a regret bound for this scheme under mild assumptions for this algorithm.

Our proof follows commonly available proof strategies for Thompson sampling.

1: Initialize a policy πa : X → R, data so-far D0 = {}, a prior over θ in f θ -P (θ * |D0) 2: for step t in {0, . . . , T-1} do 3:

θt ∼ P (θ * |Ft) (Sample θt from the posterior) 4:

Query xt = arg maxx E[f θ t (x) | θ = θt] (Query based on the posterior probability xt is optimal) 5:

Observe outcome: (xt, f (xt)) 6: Dt+1 = Dt ∪ (xt, f (xt)) 7: end for Notation The TS algorithm queries the true function f at locations (x t ) t∈N and observes true function values at these points f (x t ).

The true function f (x) is one of many possible functions that can be defined over the space R |X | .

Instead of representing the true objective function as a point object, it is common to represent a distribution p * over the true function f .

This is justified because, often, multiple parameter assignments θ, can give us the same overall function.

We parameterize f by a set of parameters θ * .

The T period regret over queries x 1 , · · · , x T is given by the random variable

Since selection of x t can be a stochastic, we analyse Bayes risk (Russo & Van Roy, 2016; Russo et al., 2018) , we define the Bayes risk as the expected regret over randomness in choosing x t , observing f (x t ), and over the prior distribution P (θ * ).

This definition is consistent with Russo & Van Roy (2016) .

Let π TS be the policy with which Thompson sampling queries new datapoints.

We do not make any assumptions on the stochasticity of π TS , therefore, it can be a stochastic policy in general.

However, we make 2 assumptions (A1, A2).

The same assumptions have been made in Russo & Van Roy (2016) .

: sup x f (x) − inf x f (x) ≤ 1 (Difference between max and min scores is bounded by 1) -If this is not true, we can scale the function values so that this becomes true.

A2: Effective size of X is finite.

1 TS (Alg 3) queries the function value at x based on the posterior probability that x is optimal.

More formally, the distribution that TS queries x t from can be written as: π TS t = P (x * = ·|D t ).

When we use parameters θ to represent the function parameter, and thus this reduces to sampling an input that is optimal with respect to the current posterior at each iteration:

MINs (Alg 2) train inverse maps f

θ (z, y), where y ∈ R. We call an inverse map optimal if it is uniformly optimal given θ t , i.e. ||f

where ε t is controllable (usually the case in supervised learning, errors can be controlled by cross-validation).

Now, we are ready to show that the regret incurred the randomized labelling active data collection scheme is bounded by O( √ T ).

Our proof follows the analysis of Thompson sampling presented in Russo & Van Roy (2016) .

We first define information ratio and then use it to prove the regret bound.

Information Ratio Russo & Van Roy (2016) related the expected regret of TS to its expected information gain i.e. the expected reduction in the entropy of the posterior distribution of X * .

Information ratio captures this quantity, and is defined as:

where I(·, ·) is the mutual information between two random variables and all expectations E t are defined to be conditioned on D t .

If the information ratio is small, Thompson sampling can only incur large regret when it is expected to gain a lot of information about which x is optimal.

Russo & Van Roy (2016) then bounded the expected regret in terms of the maximum amount of information any algorithm could expect to acquire, which they observed is at most the entropy of the prior distribution of the optimal x. Lemma C.1 (Bayes-regret of vanilla TS) (Russo & Van Roy, 2016) ).

For any T ∈ N, if Γ t ≤ Γ (i.e. information ratio is bounded above) a.s.

for each t ∈ {1, . . .

, T },

We refer the readers to the proof of Proposition 1 in Russo & Van Roy (2016) .

The proof presented in Russo & Van Roy (2016) does not rely specifically on the property that the query made by the Thompson sampling algorithm at each iteration x t is posterior optimal, but rather it suffices to have a bound on the maximum value of the information ratio Γ t at each iteration t. Thus, if an algorithm chooses to query the true function at a datapoint x t such that these queries always contribute in learning more about the optimal function, i.e. I(·, ·) appearing in the denominator of Γ is always more than a threshold, then information ratio is lower bounded, and that active data collection algorithm will have a sublinear asymptotic regret.

We are interested in the case when the active data collection algorithm queries a datapoint x t at iteration t, such that x t is the optimum for a functionfθ t , wherê θ t is a sample from the posterior distribution over θ t , i.e.θ t lies in the high confidence region of the posterior distribution over θ t given the data D t seen so far.

In this case, the mutual information between the optimal datapoint x and the observed (x t , f (x t )) input-score pair is likely to be greater than 0.

More formally,

The randomized labeling scheme for active data collection in MINs performs this step.

The algorithm samples a bunch of (x, y) datapoints, sythetically generated, -for example, in our experiments, we add noise to the values of x, and randomly pair them with unobserved or rarely observed values of y. If the underlying true function f is smooth, then there exist a finite number of points that are sufficient to uniquely describe this function f .

One measure to formally characterize this finite number of points that are needed to uniquely identify all functions in a function class is given by Eluder dimension (Russo & Van Roy) .

By augmenting synthetic datapoints and training the inverse map on this data, the MIN algorithm ensures that the inverse map is implicitly trained to be an accurate inverse for the unique function fθ t that is consistent with the set of points in the dataset D t and the augmented set S t .

Which sets of functions can this scheme represent?

The functions should be consistent with the data seen so far D t , and can take randomly distributed values outside of the seen datapoints.

This can roughly argued to be a sample from the posterior over functions, which Thompson sampling would have maintained given identical history D t .

Lemma C.2 (Bounded-error training of the posterior-optimal x t preserves asymptotic Bayes-regret).

∀t ∈ N, letx t be any input such that f (x t ) ≥ max x E[f (x)|D t ] − ε t .

If MIN chooses to query the true function atx t and if the sequence (ε t ) t∈N satisfies T t=0 ε t = O( √ T ), then, the regret from querying this ε t -optimalx t which is denoted in general as the policyπ TS is given by E[Regret(T,π

Proof.

This lemma intuitively shows that if posterior-optimal inputs x t can be "approximately" queried at each iteration, we can still maintain sublinear regret.

To see this, note:

The second term can be bounded by the absolute value in the worst case, which amounts T t=0 ε t extra Bayesian regret.

As Bayesian regret of TS is O( √ T ) and

Theorem C.3 (Bayesian Regret of randomized labeling active data collection scheme proposed in Section 3.4 is O( √ T )).

Regret incurred by the MIN algorithm with randomized labeling is of the order O( (ΓH(X * ) + C)T ).

Proof.

Simply put, we will combine the insight about the mutual information I(x , (x t , f (x t ))) > 0 and C.2 in this proof.

Non-zero mutual information indicates that we can achieve a O( √ T ) regret if we query x t s which are optimal corresponding to some implicitly defined forward function lying in the high confidence set of the true posterior given the observed datapoints D t .

Lemma C.2 says that if bounded errors are made in fitting the inverse map, the overall regret remains O( √ T ).

More formally, if ||f

and now application of Lemma C.2 gives us the extra regret incurred. (Note that this also provides us a way to choose the number of training steps for the inverse map)

Further, note if we sample x t at iteration t from a distribution that shares support with the true posterior over optimal x t (which is used by TS), we still incur sublinear, bounded O( Γ H(A * )T ) regret.

In the worst case, the overall bias caused due to the approximations will lead to an additive cumulative increase in the Bayesian regret, and hence, there is a constant

Figure 4: Contextual MBO on MNIST.

In (a) and (b), top one-half and top one-fourth of the image respectively and in (c) the one-hot encoded label are provided as contexts.

The goal is to produce the maximum stroke width character that is valid given the context.

In (a) and (b), we show triplets of the groundtruth digit (green), the context passed as input (yellow) and the produced images x from the MIN model (purple).

In this set of static dataset experiments, we study contextual MBO tasks on image pixels.

Unlike the contextual bandits case, where x corresponds to an image label, here x corresponds to entire images.

We construct several tasks.

First, we study stroke width optimization on MNIST characters, where the context is the class of the digit we wish to optimize.

Results are shown in Figure 4 .

MINs correctly produce digits of the right class, and achieve an average score over the digit classes of 237.6, whereas the average score of the digits in the dataset is 149.0.

The next task is to test the ability of MINs to be able to complete/inpaint unobserved patches of an image given an observed context patch.

We use two masks: mask A: only top half and mask B: only top one-fourth parts of the image are visible, to mask out portions of the image and present the masked image as context c to the MIN, with the goal being to produce a valid completion x, while still maximizing score corresponding to the stroke width.

We present some The aim is to maximize the score of an image which is given by the sum of attributes: eyeglasses, smiling, wavy hair and no beard.

MINs produce optimal x -visually these solutions indeed optimize the score.

sample completions in Figure 4 .

The quantitative results are presented in Table 6 .

We find that MINs are effective as compared completions for the context in the dataset in terms of score while still producing a visibly valid character.

We evaluate MINs on a complex semantic optimization task on the CelebA (Liu et al., 2015) dataset.

We choose a subset of attributes and provide their one-hot encoding as context to the model.

The score is equal to the 1 norm of the binary indicator vector for a different subset of attributes disjoint from the context.

We present our results in Figure 3 .

We observe that MINs produce diverse images consistent with the context, and is also able to effectively infer the score function, and learn features to maximize it.

Some of the model produced optimized solutions were presented in Section 4 in Figure 3 .

In this section, we present the produced generations for some other contexts.

Figure 7 shows these results.

In this section, we present some additional results for non-contextual image optimization problems.

We also evaluated our contextual optimization procedure on the CelebA dataset in a non-contextual setting.

The reward function is the same as that in the contextual setting -the sum of attributes: wavy hair, no beard, smiling and eyeglasses.

We find that MINs are able to sucessfully produce solutions in this scenario as well.

We show some optimized outputs at different iterations from the model in Figure 5 .

cGAN baseline.

We compare our MIN model to a cGAN baseline on the IMDB-Wiki faces dataset for the semantic age optimization task.

In general, we found that the cGAN model learned to ignore the score value passed as input even when trained on the entire dataset (without excluding the youngest faces) and behaved almost like a regular unconditional GAN model when queried to produce images x corresponding to the smallest age.

We suspect that this could possibly be due to the fact that age of a person doesn't have enough direct signal to guide the model to utilize it unless other tricks like reweighting proposed in Section 3.3 which explicitly enforce the model attention to datapoints of interest, are used.

We present the produced optimized x in Figure 6 .

In Figure 8 , we highlight the quantitative score values for the stroke width score function (defined as the number of pixels which have intensity more than a threshold).

Note that MINs achieve the highest value of average score while still resembling a valid digit, that stays inside the manifold of valid digits, unlike a forward model which can get high values of the score function (number of pixels turned on), but doesn't stay on the manifold of valid digits.

Figure 6 : Optimal x solutions produced by a cGAN for the youngest face optimization task on the IMDB-faces dataset.

We note that a cGAN learned to ignore the score value and produced images as an unconditional model, without any noticeable correlation with the score value.

The samples produced mostly correspond to the most frequently occurring images in the dataset.

Figure 7: Images returned by the MIN optimization for optimization over images.

We note that MINs perform successful optimization over the an objective defined by the sum of desired attributes.

Moreover, for unseen contexts, such as both brown and black hair, the optimized solutions look aligning with the context reasonably, and optimize for the score as well.

In this section, we explain the experimental details and the setup of our model.

For our experiments involving MNIST and optimization of benchmark functions task, we used the same architecture as a fully connected GAN -where the generator and discriminator are both fully connected networks.

We based our code for this part on the open-source implementation (Linder-Norén).

For the forward model experiments in these settings, we used a 3-layer feedforward ReLU network with hidden units of size 256 each in this setting.

For all experiments on CelebA and IMDB-Wiki faces, we used the VGAN (Peng et al., 2019) model and the associated codebase as our starting setup.

For experiments on batch contextual bandits, we used a fully connected discriminator and generator for MNIST, and a convolutional generator and Resnet18-like discriminator for CIFAR-10.

The prediction in this setting is categorical -1 of 10 labels needs to be predicted, so instead of using reinforce or derivative free optimization to train the inverse map, we used the Gumbel-softmax Jang et al. (2016) trick with a temperature τ = 0.75, to be able to use stochastic gradient descent to train the model.

For the protein flourescence maximization experiment, we used a 2-layer, 256-unit feed-forward gumbel-softmax inverse map and a 2-layer feed-forward discriminator.

We trained models present in open-source implementations of BanditNet (Sachdeva) , but were unable to reproduce results as reported by Joachims et al. (2018) .

Thus we reported the paper reported numbers from the BanditNet paper in the main text as well.

Temperature hyperparameter τ which is used to compute the reweighting distribution is adaptively chosen based on the 90 th percentile score in the dataset.

For example, if the difference between y max and y 90 th −percentile is given by α, we choose τ = α.

This scheme can adaptively change temperatures in the active setting.

In order to select the constant which decides whether the bin corresponding to a particular value of y is small or not, we first convert the expression Ny Ny+λ to use densities rather than absolute counts, that is,p D (y) p D (y)+λ , wherep D (y) is the empirical density of observing y in D, and now we use the same constant λ = 0.003.

We did not observe a lot of sensitivity to λ values in the range [0.0001, 0.007], all of which performed reasonably similar.

We usually fixed the number of bins to 20 for the purposed of reweighting, however note that the inverse map was still trained on continuous y values, which helps it extrapolate.

In the active setting, we train two copies of f −1 jointly side by side.

One of them is trained on the augmented datapoints generated out of the randomized labelling procedure, and the other copy is just trained on the real datapoints.

This was done so as to prevent instabilities while training inverse maps.

Training can also be made more incremental in this manner, and we need to train an inverse map to optimality inside every iteration of the active MIN algorithm, but rather we can train both the inverse maps for a fixed number of gradient steps.

<|TLDR|>

@highlight

We propose a novel approach to solve data-driven model-based optimization problems in both passive and active settings that can scale to high-dimensional input spaces.