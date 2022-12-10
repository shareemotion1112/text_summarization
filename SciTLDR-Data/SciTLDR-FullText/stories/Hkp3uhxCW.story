In this work we explore a straightforward variational Bayes scheme for Recurrent Neural Networks.

Firstly, we show that a simple adaptation of truncated backpropagation through time can yield good quality uncertainty estimates and superior regularisation at only a small extra computational cost during training, also reducing the amount of parameters by 80\%.

Secondly, we demonstrate how a novel kind of posterior approximation yields further improvements to the performance of Bayesian RNNs.

We incorporate local gradient information into the approximate posterior to sharpen it around the current batch statistics.

We show how this technique is not exclusive to recurrent neural networks and can be applied more widely to train Bayesian neural networks.

We also empirically demonstrate how Bayesian RNNs are superior to traditional RNNs on a language modelling benchmark and an image captioning task, as well as showing how each of these methods improve our model over a variety of other schemes for training them.

We also introduce a new benchmark for studying uncertainty for language models so future methods can be easily compared.

Recurrent Neural Networks (RNNs) achieve state-of-the-art performance on a wide range of sequence prediction tasks BID0 BID22 BID50 BID32 .

In this work we examine how to add uncertainty and regularisation to RNNs by means of applying Bayesian methods to training.

This approach allows the network to express uncertainty via its parameters.

At the same time, by using a prior to integrate out the parameters to average across many models during training, it gives a regularisation effect to the network.

Recent approaches either justify dropout BID43 and weight decay as a variational inference scheme BID12 , or apply Stochastic Gradient Langevin dynamics (Welling & Teh, 2011, SGLD) to truncated backpropagation in time directly BID13 .

Interestingly, recent work has not explored further directly applying a variational Bayes inference scheme BID3 for RNNs as was done in BID14 .

We derive a straightforward approach based upon Bayes by Backprop that we show works well on large scale problems.

Our strategy is a simple alteration to truncated backpropagation through time that results in an estimate of the posterior distribution on the weights of the RNN.

This formulation explicitly leads to a cost function with an information theoretic justification by means of a bits-back argument BID18 where a KL divergence acts as a regulariser.

The form of the posterior in variational inference shapes the quality of the uncertainty estimates and hence the overall performance of the model.

We shall show how performance of the RNN can be improved by means of adapting ("sharpening") the posterior locally to a batch.

This sharpening adapts the variational posterior to a batch of data using gradients based upon the batch.

This can be viewed as a hierarchical distribution, where a local batch gradient is used to adapt a global posterior, forming a local approximation for each batch.

This gives a more flexible form to the typical assumption of Gaussian posterior when variational inference is applied to neural networks, which reduces variance.

This technique can be applied more widely across other Bayesian models.

The contributions of our work are as follows:• We show how Bayes by Backprop (BBB) can be efficiently applied to RNNs.• We develop a novel technique which reduces the variance of BBB, and which can be widely adopted in other maximum likelihood frameworks.•

We improve performance on two widely studied benchmarks outperforming established regularisation techniques such as dropout by a big margin.• We introduce a new benchmark for studying uncertainty of language models.

Bayes by Backprop BID14 is a variational inference BID46 scheme for learning the posterior distribution on the weights θ ∈ R d of a neural network.

This posterior distribution is typically taken to be a Gaussian with mean parameter µ ∈ R d and standard deviation parameter σ ∈ R d , denoted N (θ|µ, σ 2 ).

Note that we use a diagonal covariance matrix, and d -the dimensionality of the parameters of the network -is typically in the order of millions.

Let log p(y|θ, x) be the log-likelihood of the model, then the network is trained by minimising the variational free energy: DISPLAYFORM0 where p(θ) is a prior on the parameters.

Minimising the variational free energy (1) is equivalent to maximising the log-likelihood log p(y|θ, x) subject to a KL complexity term on the parameters of the network that acts as a regulariser: DISPLAYFORM1 In the Gaussian case with a zero mean prior, the KL term can be seen as a form of weight decay on the mean parameters, where the rate of weight decay is automatically tuned by the standard deviation parameters of the prior and posterior.

Please refer to the supplementary material for the algorithmic details on Bayes by Backprop.

The uncertainty afforded by Bayes by Backprop trained networks has been used successfully for training feedforward models for supervised learning and to aid exploration by reinforcement learning agents BID30 BID21 , but as yet, it has not been applied to recurrent neural networks.

The core of an RNN, f , is a neural network that maps the RNN state s t at step t, and an input observation x t to a new RNN state s t+1 , f : (s t , x t ) → s t+1 .

The exact equations of an LSTM core can be found in the supplemental material Sec A.2.An RNN can be trained on a sequence of length T by backpropagation through by unrolling T times into a feedforward network.

Explicitly, we set s i = f (s i−1 , x i ), for i = 1, . . .

, T .

We shall refer to an RNN core unrolled for T steps by s 1:T = F T (x 1:T , s 0 ).

Note that the truncated version of the algorithm can be seen as taking s 0 as the last state of the previous batch, s T .RNN parameters are learnt in much the same way as in a feedforward neural network.

A loss (typically after further layers) is applied to the states s 1:T of the RNN, and then backpropagation is used to update the weights of the network.

Crucially, the weights at each of the unrolled steps are shared.

Thus each weight of the RNN core receives T gradient contributions when the RNN is unrolled for T steps.

Applying BBB to RNNs is depicted in FIG0 where the weight matrices of the RNN are drawn from a distribution (learnt by BBB).

However, this direct application raises two questions: when to sample the parameters of the RNN, and how to weight the contribution of the KL regulariser of (2).

We shall briefly justify the adaptation of BBB to RNNs, given in FIG0 .

The variational free energy of (2) for an RNN on a sequence of length T is: DISPLAYFORM0 where p(y 1:T |θ, x 1:T ) is the likelihood of a sequence produced when the states of an unrolled RNN F T are fed into an appropriate probability distribution.

The parameters of the entire network are

Sample ∼ N (0, I), ∈ R d , and set network parameters to θ = µ + σ .

Sample a minibatch of truncated sequences (x, y).

Do forward and backward propagation as normal, and let g be the gradient w.r.t θ.

DISPLAYFORM0 be the gradients of log N (θ|µ, σ 2 ) − log p(θ) w.r.t.

θ, µ and σ respectively.

Update µ using the gradient DISPLAYFORM1 Update σ using the gradient θ.

Although the RNN is unrolled T times, each weight is penalised just once by the KL term, rather than T times.

Also clear from (3) is that when a Monte Carlo approximation is taken to the expectation, the parameters θ should be held fixed throughout the entire sequence.

DISPLAYFORM2 Two complications arise to the above naive derivation in practice: firstly, sequences are often long enough and models sufficiently large, that unrolling the RNN for the whole sequence is prohibitive.

Secondly, to reduce variance in the gradients, more than one sequence is trained at a time.

Thus the typical regime for training RNNs involves training on mini-batches of truncated sequences.

Let B be the number of mini-batches and C the number of truncated sequences ("cuts"), then we can write (3) as: DISPLAYFORM3 where the (b, c) superscript denotes elements of cth truncated sequence in the bth minibatch.

Thus the free energy of mini-batch b of a truncated sequence c can be written as: DISPLAYFORM4 where w Finally, the question of when to sample weights follows naturally from taking a Monte Carlo approximations to (5): for each minibatch, sample a fresh set of parameters.

The choice of variational posterior q(θ) as described in Section 3 can be enhanced by adding side information that makes the posterior over the parameters more accurate, thus reducing variance of the learning process.

Akin to Variational Auto Encoders (VAEs) BID25 BID41 , which propose a powerful distribution q(z|x) to improve the gradient estimates of the (intractable) likelihood function p(x), here we propose a similar approach.

Namely, for a given minibatch of data (inputs and targets) (x, y) sampled from the training set, we construct such q(θ|(x, y)).

Thus, we compute a proposal distribution where the latents (z in VAEs) are the parameters θ (which we wish to integrate out), and the "privileged" information upon which we condition is a minibatch of data.

We could have chosen to condition on a single example (x, y) instead of a batch, but this would have yielded different parameter vectors θ per example.

Conditioning on the full minibatch has the advantage of producing a single θ per minibatch, so that matrix-matrix operations can still be carried.

This "sharpened" posterior yields more stable optimisation, a common pitfall of Bayesian approaches to train neural networks, and the justification of this method follows from strong empirical evidence and extensive work on VAEs.

A challenging aspect of modelling the variational posterior q(θ|(x, y)) is the large number of dimensions of θ ∈ R d .

When the dimensionality is not in the order of millions, a powerful non-linear function (such as a neural network) can be used which transforms observations (x, y) to the parameters of a Gaussian distribution, as proposed in BID25 ; BID41 .

Unfortunately, this neural network would have far too many parameters, making this approach unfeasible.

Given that the loss − log p(y|θ, x) is differentiable with respect to θ, we propose to parameterise q as a linear combination of θ and g θ = −∇ θ log p(y|θ, x), both d-dimensional vectors.

Thus, we can define a hierarchical posterior of the form DISPLAYFORM0 with µ, σ ∈ R d , and q(ϕ) = N (ϕ|µ, σ) -the same as in the standard BBB method.

Finally, let * denote element-wise multiplication, we then have DISPLAYFORM1 where η ∈ R d is a free parameter to be learnt and σ 0 a scalar hyper-parameter of our model.

η can be interpreted as a per-parameter learning rate.

During training, we get θ ∼ q(θ|(x, y)) via ancestral sampling to optimise the loss DISPLAYFORM2 DISPLAYFORM3 where µ, σ, η are our model parameters, and p are the priors for the distributions defining q (for exact details of these distributions see Section 6).

The constant C is the number of truncated sequences as defined in Section3.

The bound on the true data likelihood which yields eq. (8) is derived in Sec 4.1.

Algorithm 1 presents how learning is performed in practice.

Sample a minibatch (x, y) of truncated sequences.

DISPLAYFORM0 As long as the improvement of the log likelihood log p(y|θ, x) term along the gradient g ϕ is greater than the KL cost added for posterior sharpening (KL [q(θ|ϕ, (x, y)) || p(θ|ϕ)]), then the lower bound in (8) will improve.

This justifies the effectiveness of the posterior over the parameters proposed in eq. 7 which will be effective as long as the curvature of log p(y|θ, x) is large.

Since η is learnt, it controls the tradeoff between curvature improvement and KL loss.

Studying more powerful parameterisations is part of future research.

Unlike regular BBB where the KL terms can be ignored during inference, there are two options for doing inference under posterior sharpening.

The first involves using q(ϕ) and ignoring any KL terms, similar to regular BBB.

The second involves using q(θ|ϕ, (x, y)) which requires using the term KL [q(θ|ϕ, (x, y)) || p(θ|ϕ)] yielding an upper bound on perplexity (lower bound in log probability; see Section 4.2 for details).

This parameterisation involves computing an extra gradient and incurs a penalty in training speed.

A comparison of the two inference methods is provided in Section 6.

Furthermore, in the case of RNNs, the exact gradient cannot be efficiently computed, so BPTT is used.

Here we turn to deriving the training loss function we use for posterior sharpening.

The basic idea is to take a variational approximation to the marginal likelihood p(x) that factorises hierarchically.

Hierarchical variational schemes for topic models have been studied previously in BID40 .

Here, we shall assume a hierarchical prior for the parameters such that p(x) = p(x|θ)p(θ|ϕ)p(ϕ)dθdϕ. Then we pick a variational posterior that conditions upon x, and factorises as q(θ, ϕ|x) = q(θ|ϕ, x)q(ϕ).

The expected lower bound on p(x) is then as follows: DISPLAYFORM0 DISPLAYFORM1

We note that the procedure of sharpening the posterior as explained above has similarities with other techniques.

Perhaps the most obvious one is line search: indeed, η is a trained parameter that does line search along the gradient direction.

Probabilistic interpretations have been given to line search in e.g. BID34 , but ours is the first that uses a variational posterior with the reparametrization trick/perturbation analysis gradient.

Also, the probabilistic treatment to line search can also be interpreted as a trust region method.

Another related technique is dynamic evaluation BID37 , which trains an RNN during evaluation of the model with a fixed learning rate.

The update applied in this case is cumulative, and only uses previously seen data.

Thus, they can take a purely deterministic approach and ignore any KL between a posterior with privileged information and a prior.

As we will show in Section 6, performance gains can be significant as the data exhibits many short term correlations.

Lastly, learning to optimise (or learning to learn) BID28 BID1 is related in that a learning rate is learned so that it produces better updates than those provided by e.g. AdaGrad BID9 or Adam BID24 .

Whilst they train a parametric model, we treat these as free parameters (so that they can adapt more quickly to the non-stationary distribution w.r.t.

parameters).

Notably, we use gradient information to inform a variational posterior so as to reduce variance of Bayesian Neural Networks.

Thus, although similar in flavour, the underlying motivations are quite different.

Applying Bayesian methods to neural networks has a long history, with most common approximations having been tried.

BID5 propose various maximum a posteriori schemes for neural networks, including an approximate posterior centered at the mode.

Buntine & Weigend (1991) also suggest using second order derivatives in the prior to encourage smoothness of the resulting network.

BID18 proposed using variational methods for compressing the weights of neural networks as a regulariser.

BID20 suggest an MDL loss for single layer networks that penalises non-robust weights by means of an approximate penalty based upon perturbations of the weights on the outputs.

Denker & Lecun (1991); MacKay (1995) investigated using the Laplace approximation for capturing the posterior of neural networks.

Neal (2012) investigated the use of hybrid Monte Carlo for training neural networks, although it has so far been difficult to apply these to the large sizes of networks considered here.

More recently Graves (2011) derived a variational inference scheme for neural networks and Blundell et al. FORMULA0 extended this with an update for the variance that is unbiased and simpler to compute.

Graves (2016) derives a similar algorithm in the case of a mixture posterior.

Several authors have claimed that dropout BID43 and Gaussian dropout BID47 can be viewed as approximate variational inference schemes BID11 Kingma et al., 2015, respectively FORMULA0 proposed a variational scheme with biased gradients for the variance parameter using the Fisher matrix.

Our work extends this by using an unbiased gradient estimator without need for approximating the Fisher and also add a novel posterior approximation.

Variational methods typically underestimate the uncertainty in the posterior (as they are mode seeking, akin to the Laplace approximation), whereas expectation propagation methods often average over modes and so tend to overestimate uncertainty (although there are counter examples for each depending upon the particular factorisation and approximations used; see for example BID44 ).

Nonetheless, several papers explore applying expectation propagation to neural networks: BID42 derive a closed form approximate online expectation propagation algorithm, whereas Hernández-Lobato & Adams (2015) proposed using multiple passes of assumed density filtering (in combination with early stopping) attaining good performance on a number of small data sets.

BID16 derive a distributed expectation propagation scheme with SGLD BID48 as an inner loop.

Others have also considered applying SGLD to neural networks BID27 and BID13 more recently used SGLD for LSTMs (we compare to these results in our experiments).

We present the results of our method for a language modelling and an image caption generation task.

We evaluated our model on the Penn Treebank BID35 benchmark, a task consisting on next word prediction.

We used the network architecture from BID50 , a simple yet strong baseline on this task, and for which there is an open source implementation 1 .

The baseline consists of an RNN with LSTM cells and a special regularisation technique, where the dropout operator is only applied to the non-recurrent connections.

We keep the network configuration unchanged, but instead of using dropout we apply our Bayes by Backprop formulation.

Our goal is to demonstrate the effect of applying BBB to a pre-existing, well studied architecture.

To train our models, we tuned the parameters on the prior distribution, the learning rate and its decay.

The weights were initialised randomly and we used gradient descent with gradient clipping for optimisation, closely following BID50 's "medium" LSTM configuration (2 layers with 650 units each).

As in , the prior of the network weights θ was taken to be a scalar mixture of two Gaussian densities with zero mean and variances σ 2 1 and σ 2 2 , explicitly DISPLAYFORM0 where θ j is the j-th weight of the network.

We searched π ∈ {0.25, 0.5, 0.75}, log σ 1 ∈ {0, −1, −2} and log σ 2 ∈ {−6, −7, −8}.For speed purposes, during training we used one sample from the posterior for estimating the gradients and computing the (approximate) KL-divergence.

For prediction, we experimented with either computing the expected loss via Monte Carlo sampling, or using the mean of the posterior distribution as the parameters of the network (MAP estimate).

We observed that the results improved as we increased the number of samples but they were not significantly better than taking the mean (as was also reported by BID14 ).

For convenience, in TAB2 we report our numbers using the mean of the converged distribution, as there is no computation overhead w.r.t.

a standard LSTM model.

TAB2 compares our results to the LSTM dropout baseline BID50 we built from, and to the Variational LSTMs BID12 , which is another Bayesian approach to this task.

Finally, we added dynamic evaluation BID37 results with a learning rate of 0.1, which was found via cross validation.

As with other VAE-related RNNs BID10 ; BID2 BID7 perplexities using posterior sharpening are reported including a KL penalty KL [q(θ|ϕ, (x, y)) || p(θ|ϕ)] in the log likelihood term (the KL is computed exactly, not sampled).

For posterior sharpening we use a hierarchical prior for θ: p(θ|ϕ) = N (θ|ϕ, σ 2 0 I) which expresses our belief that a priori, the network parameters θ will be much like the data independent parameters ϕ with some small Gaussian perturbation.

In our experiments we swept over σ 0 on the validation set, and found σ 0 = 0.02 to perform well, although results were not particularly sensitive to this.

Note that with posterior sharpening, the perplexities reported are upper bounds (as the likelihoods are lower bounds).Lastly, we tested the variance reduction capabilities of posterior sharpening by analysing the perplexity attained by the best models reported in TAB2 .

Standard BBB yields 258 perplexity after only one epoch, whereas the model with posterior sharpening is better at 227.

We also implemented it on MNIST following , and obtained small but consistent speed ups.

Lower perplexities on the Penn Treebank task can be achieved by varying the model architecture, which should be complementary to our work of treating weights as random variables-we are simply interested in assessing the impact of our method on an existing architecture, rather than absolute state-of-the-art.

See BID23 ; Zilly et al. FORMULA0 ; BID36 , for a report on recent advances on this benchmark, where they achieve perplexities of 70.9 on the test set.

Furthermore we note that the speed of our naïve implementation of Bayesian RNNs was 0.7 times the original speed and 0.4 times the original speed for posterior sharpening.

Notably, FIG2 shows the effect of weight pruning: weights were ordered by their signal-to-noise ratio (|µ i |/σ i ) and removed (set to zero) in reverse order.

We evaluated the validation set perplexity for each proportion of weights dropped.

As can be seen, around 80% of the weights can be removed from the network with little impact on validation perplexity.

Additional analysis on the existing patterns of the dropped weights can be found in the supplementary material A.3.

We used the Penn Treebank test set, which is a long sequence of ≈ 80K words, and reversed it.

Thus, the "reversed" test set first few words are: "us with here them see not may we ..." which correspond to the last words of the standard test set: "... we may not see them here with us".Let V be the vocabulary of this task.

For a given input sequence x = x 1:T and a probabilistic model p, we define the entropy of x under p, H p [x] , by DISPLAYFORM0 DISPLAYFORM1 , i.e., the per word entropy.

Let X be the standard Penn Treebank test set, and X rev the reversed one.

For a given probabilistic model p, we define the entropy gap ∆H p by DISPLAYFORM2 Since X rev clearly does not come from the training data distribution (reversed English does not look like proper English), we expect ∆H p to be positive and large.

Namely, if we take the per word entropy of a model as a proxy for the models' certainty (low entropy means the model is confident about its prediction), then the overall certainty of well calibrated models over X rev should be lower than over X. Thus, DISPLAYFORM3 .

When comparing two distributions, we expect the better calibrated one to have a larger ∆H p .In FIG3 , we plotted ∆H p for the BBB and the baseline dropout LSTM model.

The BBB model has a gap of about 0.67 nats/word when taking 10 samples, and slightly below 0.65 when using the posterior mean.

In contrast, the model using MC Dropout BID11 is less well calibrated and is below 0.58 nats/word.

However, when "turning off" dropout (i.e., using the mean field approximation), ∆H p improves to below 0.62 nats/word.

We note that with the empirical likelihood of the words in the test set with size T (where for each word w ∈ V , p(w) = (# occurrences of w) T ), we get an entropy of 6.33 nats/word.

The BBB mean model has entropy of 4.48 nats/word on the reversed set which is still far below the entropy we get by using the empirical likelihood distribution.

We also applied Bayes by Backprop for RNNs to image captioning.

Our experiments were based upon the model described in , where a state-of-the-art pre-trained convolutional neural network (CNN) was used to map an image to a high dimensional space, and this representation was taken to be the initial state of an LSTM.

The LSTM model was trained to predict the next word on a sentence conditioned on the image representation and all the previous words in the image caption.

We kept the CNN architecture unchanged, and used an LSTM trained using Bayes by Backprop rather than the traditional LSTM with dropout regularisation.

As in the case for language modelling, this work conveniently provides an open source implementation 2 .

We used the same prior distribution on the weights of the network (18) as we did for the language modelling task, and searched over the same hyper-parameters.

We used the MSCOCO BID29 ) data set and report perplexity, BLUE-4, and CIDER scores on compared to the Show and Tell model We observe significant improvements in BLUE and CIDER, outperforming the dropout baseline by a large margin.

Moreover, a random sample of the captions that were different for both the baseline and BBB is shown in FIG4 .

Besides the clear quantitative improvement, it is useful to visualise qualitatively the performance of BBB, which indeed generally outperforms the strong baseline, winning in most cases.

As in the case of Penn Treebank, we chose a performant, open source model.

Captioning models that use spatial attention, combined with losses that optimise CIDER directly (rather than a surrogate loss as we do) achieve over 100 CIDER points BID32 BID31 .

We have shown how to apply the Bayes by Backprop (BBB) technique to RNNs.

We enhanced it further by introducing the idea of posterior sharpening: a hierarchical posterior on the weights of neural networks that allows a network to adapt locally to batches of data by a gradient of the model.

We showed improvements over two open source, widely available models in the language modelling and image captioning domains.

We demonstrated that not only do BBB RNNs often have superior performance to their corresponding baseline model, but are also better regularised and have superior uncertainty properties in terms of uncertainty on out-of-distribution data.

Furthermore, BBB RNNs through their uncertainty estimates show signs of knowing what they know, and when they do not, a critical property for many real world applications such as self-driving cars, healthcare, game playing, and robotics.

Everything from our work can be applied on top of other enhancements to RNN/LSTM models (and other non-recurrent architectures), and the empirical evidence combined with improvements such as posterior sharpening makes variational Bayes methods look very promising.

We are exploring further research directions and wider adoption of the techniques presented in our work.

The core of an RNN, f , is a neural network that maps the RNN state at step t, s t and an input observation x t to a new RNN state s t+1 , f : (s t , x t ) → s t+1 .An LSTM core Hochreiter & Schmidhuber (1997) has a state s t = (c t , h t ) where c is an internal core state and h is the exposed state.

Intermediate gates modulate the effect of the inputs on the outputs, namely the input gate i t , forget gate f t and output gate o t .

The relationship between the inputs, outputs and internal gates of an LSTM cell (without peephole connections) are as follows: As discussed in Section 6.1, for the Penn Treebank task, we have taken the converged model an performed weight pruning on the parameters of the network.

Weights were ordered by their signalto-noise ratio (|µ i |/σ i ) and removed (set to zero) in reverse order.

It was observed that around 80% of the weights can be removed from the network with little impact on validation perplexity.

In Figure 5 , we show the patterns of the weights dropped for one of the LSTM cells from the model.

DISPLAYFORM0

Figure 5: Pruning patterns for one LSTM cell (with 650 untis) from converged model with 80% of total weights dropped.

A white dot indicates that particular parameter was dropped.

In the middle column, a horizontal white line means that row was set to zero.

Finally, the last column indicates the total number of weights removed for each row.

@highlight

 Variational Bayes scheme for Recurrent Neural Networks