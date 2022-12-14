Generative neural networks map a standard, possibly distribution to a complex high-dimensional distribution, which represents the real world data set.

However, a determinate input distribution as well as a specific architecture of neural networks may impose limitations on capturing the diversity in the high dimensional target space.

To resolve this difficulty, we propose a training framework that greedily produce a series of generative adversarial networks that incrementally capture the diversity of the target space.

We show theoretically and empirically that our training algorithm converges to the theoretically optimal distribution, the projection of the real distribution onto the convex hull of the network's distribution space.

Generative Adversarial Nets (GAN) BID5 is a framework of estimating generative models.

The main idea BID4 is to train two target network models simultaneously, in which one, called the generator, aims to generate samples that resemble those from the data distribution, while the other, called the discriminator, aims to distinguish the samples by the generator from the real data.

Naturally, this type of training framework admits a nice interpretation as a twoperson zero-sum game and interesting game theoretical properties, such as uniqueness of the optimal solution, have been derived BID5 .

It is further proved that such adversarial process minimizes certain divergences, such as Shannon divergence, between the generated distribution and the data distribution.

Simply put, the goal of training a GAN is to search for a distribution in the range of the generator that best approximates the data distribution.

The range is often defined by the input latent variable z and its specific architecture, i.e., Π = {G(z, θ), θ ∈ Θ}. When the range is general enough, one could possibly find the real data distribution.

However, in practice, the range is usually insufficient to perfectly describe the real data, which is typically of high dimension.

As a result, what we search for is in fact the I-projection BID2 of the real data distribution on Π.

1.

The range of the generator Π is convex (see figure 1(a)), or it is not convex but the projection of the real data distribution on Π's convex hull (CONV Π) is in Π (see FIG0 ).

2.

The range of the generator is non-convex and the projection of the real data distribution in CONV Π is not in the range Π (see figure 1(c)).

In case 1, one can find the optimal distribution in Π to approximate real data set in CONV Π. But in case 2, using standard GANs with a single generator, one can only find the distribution in Π that is nearest to the projection.

It then makes sense to train multiple generators and use a convex combination of them to better approximate the data distribution (than using a single generator in the non-convex case (see figure 1(c))).The above argument is based on the assumption that one could achieve global optimality by training, while this is not the case in general.

When reaching a local optimal distribution, in order to improve performance, do we need to add more generators and restart training?

In this paper, we put forward a sequential training procedure that adds generators one by one to improve the performance, without retraining the previously added generators.

Our contributions can be summarized as follows.• We derive an objective function tailored for such a incremental training process.

The objective function takes both the real data distribution and the pre-learned distribution into consideration.

We show that with this new objective, we actually maximize marginal contribution when adding a new generator.

We also put forward an incremental training algorithm based on the new objective function.• We prove that our algorithm always converges to the projection of real data distribution to the convex hull of the ranges of generators, which is the optimal solution with multiple generators.

This property continues to hold in online settings where target distribution changes dynamically.• Our experiments show that our algorithm can overcome the local optimal issue mentioned above.

We perform experiments on a synthetic dataset as well as two real world datasets, e.g., CelebA and MNIST, and conclude that our algorithm could improve the mixture distribution even in the case where the range is not sufficient enough.• Experiments also show that, compared with previous methods, our algorithm is fast and stable in reducing the divergence between mixture distribution and the real data.

Recently, there have been intensive researches on improving the performance of generative adversarial neural networks.

Two lines of works are closely related to our paper.

They focus mainly on improving the discriminator and the generator respectively.

The Unrolled GAN introduced by BID11 improves the discriminator by unrolling optimizing the objective during training, which stabilizes training and effectively reduces the mode collapse.

D2GAN proposed by utilizes two discriminators to minimize the KL-divergence and the reverse KL-divergence respectively.

It treats different modes more fairly, and thus avoids mode collapse.

DFM introduced by Warde-Farley & Bengio (2016) brings a Denoising AutoEncoder (DAE) into the generator's objective to minimize the reconstruction error in order to get more information from the target manifold.

BID12 proposed McGan based on mean and covariance feature matching to stabilize the training of GANs.

Finally, WGAN introduced by employs the Wasserstein distance, which is a more appropriate measure of performance, and achieves more stable performance.

These works are different from ours since they focus on the discriminator by measuring the divergence between the generated data and the real data more precisely.

However, our work fixes the discriminator and tries to enrich the expressiveness of the generator by combining multiple generators.1.1.2 IMPROVING THE GENERATOR Wang et al. (2016) proposes two methods to improve the training process.

The first is selfensembling GANs, which assembles the generators from different epochs to stabilize training.

The other is Cascade GAN, where the authors train new generator using the data points with highest values from the discriminator.

These two methods are heuristic ways to improve training, but with no theoretical guarantee.

BID7 and BID3 proposed the methods called MGAN and multi-agent GANs respectively.

The former introduces a classifier into the discriminator to catch different modes, while the later employ a new component into the generators' objective to promote diversity.

BID1 introduces a new metric on distributions and proposes a MIX+GAN to search for an equilibrium.

But all these methods need to train the multiple generators simultaneously, and none of them can deal with the case when the training process reaches a local optima.

Also, these models lack flexibility, in the sense that when one tries to change the number of generators, all the generators need to be retrained.

Another closely related work is BID15 , in which the authors propose a method called AdaGAN, which is based on a robust reweighting scheme on the data set inspired from boosting.

The idea is that the new generators should focus more on the previous bad training data.

But AdaGAN and other boosting-like algorithms are based on the assumption that one generator could catch some modes precisely, which may not be reasonable since the generator always learns to generate the average samples among the real data set in order to obtain low divergence, especially when the generator's range is under condition of FIG0 .

In Section 5, we compare our algorithm with AdaGAN with different dataset.

A GAN BID5 takes samples (a.k.a.

latent variables z) from a simple and standard distribution as its input and generates samples in a high dimensional space to approximate the target distribution.

This is done by training a generative neural network and an auxiliary discriminative neural network alternatively.

An f- BID14 generalizes the adversarial training as minimizing the f-divergence between the real data distribution and the generated distribution, DISPLAYFORM0 dx.

A GAN is a special f-GAN that minimizes the JensenShannon divergence.

The general objective function of an f-GAN can be defined as follows: min DISPLAYFORM1 Here f * is the conjugate function of f in f-divergence; T represents a neural network regarded as the corresponding discriminator; finally, θ and ξ denote the parameters of the generator and the discriminator, respectively.

The adversarial training method proposed by BID5 is playing a minimax game between the generator and the discriminator.

Such a method can be caught in local optima and thus is undesirable (e.g., mode collapse).In this paper we propose a novel framework to train multiple generators sequentially: We maintain a group of generators (empty at the beginning) as well as their corresponding weights, then add new generators into the group one by one and rebalance the weights.

In particular, only the newly added generator at each step is trained.

The purpose here is to augment the capacity of the group of generators and mitigate the local optima issue.

Define the distribution range of a generator as Π = {p | p = G(z, θ), θ ∈ Θ}, i.e., the set of distributions that the generator can produce with different parameter θ.

The distribution range is determined by the distribution of input z and the architecture of the generative network.

Define a generator group as G = {G 1 , G 2 , , . . .

, G n } , where G i is the generator added in step i. We associate each generator with a weight ω i > 0.

Then the mixed distribution of the group is: DISPLAYFORM0 ω i is the sum of weights.

When a new generator G n+1 joins in the group G, the group becomes G = G ∪ {G n+1 } and the mixed distribution becomes DISPLAYFORM1

In this section, we describe how we use a generator group to improve the performance and tackle the local optima issue mentioned previously.

To train such a generator group, we propose an incremental training algorithm (algorithm 1) adding generators to the group sequentially.

In algorithm 1, we use DISPLAYFORM0 repeat Build and initialize generator G i using the same network structure.

Set target distribution for G i to be p target = DISPLAYFORM1 until Convergence D(·, ·) to denote the "distance" between two distributions, which can be any divergence (e.g., fdivergence or Wasserstein distance) or a general norm.

The key step in algorithm 1 is the choice of the target distribution for training DISPLAYFORM2 j=1 ω j p j and after adding G i , the generator group G can perfectly produce the desired distribution p real .

However, in general, we have D(p target , p i ) = 0 and our algorithm proceeds in a greedy fashion, i.e., it always maximizes the marginal contribution of G i to the generator group G. We devote the rest of this section to proving the above statement.

In algorithm 1, we use different loss functions for each generators.

The marginal contribution of the (N + 1)-th generator is as follows when we adopt f-divergence as the distance measure: DISPLAYFORM0 To get a better approximation to the real distribution, we fix the existing generators in the group and tune the parameters of the new generator to minimize the distance between the new group and the real distribution.

In fact, this is equivalent to maximizing the marginal contribution of the new generator DISPLAYFORM1 DISPLAYFORM2 .To show this, we first introduce the χ 2 -divergence.

DISPLAYFORM3 dx.

Note that χ 2 -divergence is a special case of the f-divergence: DISPLAYFORM4 In fact, with some mild assumptions on f , the f -divergence is well-approximated by χ 2 -divergence when p and q are close.

The following lemma can be obtained via Taylor expansion BID2 .

Lemma 1.

For any f-divergence with f (u), if f (u) is twice differentiable at u = 1 and f (1) > 0, then for any q and p close to q we have: DISPLAYFORM5 Proof of proposition 1.

We rewrite the objective function equation 1 for χ 2 -divergence: DISPLAYFORM6 Based on the former definition, we obtain DISPLAYFORM7 + , which concludes the proof.

According to algorithm 1, in each round, a new generator G N +1 is added and the loss function is set to be D(p target , p N +1 ).

Therefore, when training each generator G i , the target distribution only depends on the real distribution and the previous generators in G. In particular, both of them are already known (figure 2).To minimize D(p target , p G N +1 ), we conduct adversarial training by using an auxiliary discriminator T: DISPLAYFORM0 where by the linearity of expectation: DISPLAYFORM1 .Based on these, we propose an incremental training algorithm for G N +1 as algorithm 2.

In this section, we show that although our framework which trains each generator in a greedy way, the output distribution of the generator group will always converge.

Furthermore, the converged distribution is the closest one to the target distribution among the set of all possible distributions that a group of generators can produce (i.e., the optimal one within the distribution range of the group of generators).Recall our notation that the distribution range of a generator is Π. By taking a convex combination of multiple generators (with the same architecture), the set of all possible output distributions becomes the convex hull of Π: DISPLAYFORM0

Our algorithm greedily optimizes each G N +1 to minimize D(p target , p G N +1 ).

By the Pinsker's inequality, the total variation distance between p target and p G N +1 is upper bounded by D(p target , p G N +1 )/2 and we can easily extend it to χ 2 -divergence by D KL (p||q) ≤ D χ 2 (p||q) + 0.42.

In other words, while greedily optimizing each G N +1 , the distance between p target and p G N +1 is also approximately minimized.

Hence it is reasonable to assume that for each G N +1 , its distance to p target is approximately minimized with some tolerance ≥ 0, i.e., p G N +1 −p target ≤ inf p − p target + .

Under such an assumption, our algorithm approximately converges to the the optimal distribution in CONV Π: Proposition 2.

For any Π that is connected and bounded, algorithm 2 approximately converges to the optimal distribution within the closure of the convex hull CONV Π of Π.To simplify the argument, we fix each ω i to be 1 and embed the discrete probability distributions into a Hilbert space.

In this case, each G N +1 approximately minimizes the distance to p target = (N + 1)p real − N i=1 p Gi can be formalized as: DISPLAYFORM0 and our algorithm approximately converges to the optimal distribution in CONV Π if as N → ∞, DISPLAYFORM1 Then proposition 2 is implied by the following lemma.

Lemma 2.

Consider a connected and bounded subset Π of a Hilbert space H and any target ρ ∈ H. Let {p * n } ∞ n=1 be a sequence of points in Π such that for ρ target = (n + 1)ρ − nT n , DISPLAYFORM2 Corollary 1.

With the finite change of target distribution, algorithm 2 can converge to the new optimal distribution within CONV Π.Due to the space limit, we send the proof to the appendix.

Based on corollary 1, regardless the change of target distribution, as long as it is an finite variation, algorithm 2 can converge to the projection of new target distribution.

Due to the sequential nature and the above theoretical guarantee, our algorithm naturally generalizes the dynamic online settings.

We test our algorithm on a synthesized Gaussian distribution dataset and two well-known real world datasets: CelebA and MNIST, which are the complex high dimensional real world distributions.

We design the experiment to test our sequential training algorithm.

The main purpose is not to demonstrate high quality results, e.g., high definition pictures, but to show that our algorithm can search for an appropriate distribution that significantly improved the performance of mixture distributions as the number of generators increase, especially when the generator's range is rather limited.

In all experiments, we use the Adam optimizer Kingma & Ba (2014) with learning rate of 5 × 10 −5 , and β 1 = 0.5, β 2 = 0.9.

Finally, we set weights ω i = 1 for convenience.

Metric.

As the method mentioned in BID14 , when we fix the generator, we can train an auxiliary neural network to maximize the derived lower bound to measure the divergence between the generated data and the real data, i.e., D f (P ||Q).

Based on these theories, we import an auxiliary neural network to measure the performance of different methods.

The architecture of the auxiliary neural network is the same as the discriminator used in each experiment.

We train it for 50 epoches, which is enough to measure the differences.

Then we take the mean value of the last 100 iterations as the final output.

Synthesized data.

In this part, we design some experiments in R 2 space.

The dataset is sampled from 8 independent two-dimensional Gaussian distributions (i.e., the blue points in FIG7 .

The model is previously proposed by BID11 .Firstly, following the experiment designed in BID11 and BID7 , we choose the latent variable z in a high dimensional space as z ∼ N (0, I 256 ), i.e., the distribution projection is likely to be in the generator's range, which meets the condition of FIG0 .

In FIG7 , the blue points are the real data while the corresponding colored number represents the data points generated by each generator respectively.

As FIG7 shows, we train up to 4 generators to approximate the data distribution and the first generator tends to catch the data with high probability around the centre of each Gaussian.

As the number of generators increasing, generated data tends to cover the data away from the centre in order to be complementary to previous mixture distributions and thus gains a considerable marginal profit.

These results demonstrate our marginal maximization algorithm can promote the mixture distributions to cover the data with low probabilities.

Secondly, we reduce the dimension of z to 1, i.e., z ∼ N (0, 1) and simplify the corresponding network architecture, so that the condition of figure 1(c) is likely met.

In this part, we compare our algorithm with the state of the art incremental training method AdaGAN BID15 and the baseline method Orignal GAN.

1 We train up to 20 generators in each experiment with the same starting generator (i.e., identical first generator for each method), then measure the D χ 2 (p||q) between real distribution and the generated mixed distribution.

We repeat the experiment for 30 times to reduce the effect of random noises.

Figure 5 and figure 6 illustrate the average and the best performance with different numbers of generators, respectively.

According to the result, our algorithm approaches to p real faster than the other two methods and achieves the best performance among all three methods.

In summary, our algorithm outperforms the other two both in terms of the speed of converging to the real distribution and the quality of the final result under the case of figure 1(c).MNIST.

In this experiment, we run our algorithm on the MNIST dataset BID9 .

We design this experiment to measure the performance of our algorithm for a more complex data distribution.

We choose the latent variable as z ∼ N (0, 1) to limit the corresponding generator range and Then we train up to 22 generators to approximate the real distribution and the result is showed in figure 4 .

Our algorithm outperforms the Original GAN but is inferior to the AdaGAN with the first 8 generators.

As the number of generators increases, AdaGAN seems to run into a bottleneck while both our algorithm and the Original GAN gradually approximate to the real data distribution.

In order to analysis the convergence, we further train up to 100 generators with both our algorithm and the original GAN.

In FIG4 , the horizontal dash lines represent the minimum value of the Wasserstein distance for the two method respectively.

As showed in figure, the distance gradually decrease with the number of generators increasing, and our algorithm is much faster to reduce the distance and can even obtain a better performance.

More over, as we tends to investigate the property of each generator in the generators' group, we measure the Wasserstein distance between distribution G(z, θ) and the real data, the experiment result is showed in FIG5 and the dash lines in FIG5 represent the mean value of the 100 generators.

Interestingly, the result shows that, in each generator, original GAN tends to search a distribution in the distribution range that is closer to the real data distribution, while our algorithm is searching for a distribution that is complementary to the generators' group (i.e., a huge decrease in the mixture condition in FIG4 ) even if its own performance is poor (i.e., a high distance in the in FIG5 ).CelebA.

We also conduct our experiment on the CelebA dataset BID10 .

As shown in figure 1, we start with an identical generator and train up to 6 generators using different methods.

The measured Wasserstein distance between mixed distribution of Group G and the real-data distribution is showed in FIG0 .

In this experiment, we use the training method WGAN-GP proposed by BID6 .

The experiment results indicates that our algorithm outperforms the other two methods after the second generator.

It demonstrates the potential of our algorithm applying to real world datasets.

Proof of lemma 2.

Without loss of generality, we can assume ρ = 0, since otherwise we can add an offset −ρ to the Hilbert space H. DISPLAYFORM0 Letp ∈ Π be the point that minimizes the distance between −nT n and its projectionp ⊥ on line −nT n , i.e.,p = arg min p∈Π p ⊥ + nT n , where p ⊥ = p,nTn H nTn 2 · nT n .

Then we can further bound d n+1 by p + nT n , DISPLAYFORM1 On one hand, since T n ∈ CONV Π,p ⊥ can be seen as the projection ofp on T n as well, hence p −p ⊥ ≤ p − T n .

Note that Π is bounded, therefore p − T n is bounded by the diameter of Π, denoted as d ≥ 0.On the other hand, suppose that p * is the point closest to ρ = 0 within CONV Π, i.e., p * = arg min p∈CONV Π p .

Since Π is connected, therefore the projection of Π on −nT n is the same with the projection of CONV Π. Hence, DISPLAYFORM2 In other words, DISPLAYFORM3 If p * = inf p∈CONV Π p > 0, then we have d n = n T n ≥ n p * .

Hence DISPLAYFORM4 2 /2(n + 1) p * ≤ ( p * + )(n + 1) + d 2 /2 p * · ln(n + 1).Then for any δ > 0, let N be sufficiantly large such that ln N N ≤ 2δ p * /d 2 , we have DISPLAYFORM5 Otherwise, p * = 0 and d n+1 ≤ + d 2 + d 2 n .

Note that the upper bound is increasing in d n , hence ∀n > 0, d n ≤ d * n for d * n defined as follows: DISPLAYFORM6 For which, we can easily prove by induction that d * n ≤ n + √ nd.

Therefore T n = d n ≤ + d/ √ n, which immediately completes the proof.

Proof of corollary 1.

Without loss of generality, we assume the optimal projection of target distribution is changed from ρ to ρ after n 0 iterations, where n 0 ∈ R + is a constant value.

Then we can derive T n+n0 − ρ ≤ n· Tn−ρ n+n0 DISPLAYFORM0 , where n ∈ R + is the training iteration after change.

Then based on lemma 2, we obtain lim n→+∞ T n − ρ ≤ inf p∈Π p − ρ + .

On the other side, for a specific n 0 , T n0 − ρ ≤ T n0 − ρ + ρ − ρ is a bounded value if the variation of target distribution is limited.

Finally, we can obtain lim n→+∞ T n+n0 − ρ ≤ inf p∈Π p − ρ + , which concludes the proof.

dx.

For KL-divergence and χ 2 -divergence, the corresponding f (t) are f KL (t) = t log(t) and f χ 2 (t) = (t−1) 2 respectively.

Import an auxiliary function as: DISPLAYFORM1 Then based on the monotonicity of F(t), we have F (t) min ≥ −0.42.

DISPLAYFORM2

<|TLDR|>

@highlight

We propose a new method to incrementally train a mixture generative model to approximate the information projection of the real data distribution.