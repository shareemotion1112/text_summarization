Generating complex discrete distributions remains as one of the challenging problems in machine learning.

Existing techniques for generating complex distributions with high degrees of freedom depend on standard generative models like Generative Adversarial Networks (GAN), Wasserstein GAN, and associated variations.

Such models are based on an optimization involving the distance between two continuous distributions.

We introduce a Discrete Wasserstein GAN (DWGAN) model which is based on a dual formulation of the Wasserstein distance between two discrete distributions.

We derive a novel training algorithm and corresponding network architecture based on the formulation.

Experimental results are provided for both synthetic discrete data, and real discretized data from MNIST handwritten digits.

Generative Adversarial Networks (GAN) BID3 have gained significant attention in the field of machine learning.

The goal of GAN models is to learn how to generate data based on a collection of training samples.

The GAN provides a unique training procedure by treating the learning optimization as a two player game between a generator network and discriminator network.

Since the learning process involves optimization over two different networks simultaneously, the GAN is hard to train, often times unstable BID11 .

Newly developed models such as the Wasserstein GAN aim to improve the training process by leveraging the Wasserstein distance in optimization, as opposed to the Kullback-Leibler or Jensen-Shannon divergences utilized by the original GAN.A source of interest in generative models arises from natural language processing.

In natural language applications, a generative model is necessary to learn complex distributions of text documents.

Although both the GAN and Wasserstein GAN approximate a distance between two continuous distributions, and use a continuous sample distance, prior research efforts BID4 BID12 BID10 have applied the models to discrete probability distributions advocating for a few modifications.

However, using a continuous sample distance for the discrete case may lead to discrepancies.

More precisely, as will be demonstrated via explicit examples, a small continuous distance does not necessarily imply a small discrete distance.

This observation has potentially serious ramifications for generating accurate natural language text and sentences using GAN models.

To address the above issues, we propose a Discrete Wasserstein GAN (DWGAN) which is directly based on a dual formulation of the Wasserstein distance between two discrete distributions.

A principal challenge is to enforce the dual constraints in the corresponding optimization.

We derive a novel training algorithm and corresponding network architecture as one possible solution.

Generative Adversarial Networks (GANs) BID3 ) model a sample generating distribution by viewing the problem as a two-player game between a generator and a discriminator which is an adversary.

The generator takes an input from a random distribution p(z) over a latent variable z, and maps it to the space of data x. The discriminator takes inputs from real data and Table 2 : Example of a large gap between discrete and continuous distances for a discrete sample with 9 classes.

Training sample: samples from the generator, and attempts to distinguish between the real and generated samples.

Formally, the GAN plays the following two player minimax game: DISPLAYFORM0 DISPLAYFORM1 where D is the discriminator network and G is the generator network.

In theory, the GAN approximates the Jensen-Shannon divergence (JSD) between the generated and real data distribution.

showed that several divergence metrics including the JSD do not always provide usable gradients.

Therefore, optimization based on JSD minimization, as incorporated in the GAN, will not converge in certain cases.

To overcome the problem, proposed the Wasserstein GAN which is an approximation to the dual problem of the Wasserstein distance.

The authors showed that the Wasserstein distance provides sufficient gradients almost everywhere, and is more robust for training purposes.

The dual problem of the Wasserstein distance involves an optimization over all 1-Lipschitz functions BID13 .

The Wasserstein GAN approximates the dual problem by clipping all network weights to ensure that the network represents a k-Lipschitz function for some value of k. A recent variant of the Wasserstein GAN BID4 enforced the k-Lipschitz property by adding a gradient penalty to the optimization.

Although the formulation of the Wasserstein GAN approximates the Wasserstein distance between two continuous distributions, using a continuous sample distance x − y , existing research efforts BID4 BID12 BID10 have directly used it to model discrete probability distributions by adding the following modifications.

Each component of the input vectors of training data is encoded in a one-hot representation.

A softmax nonlinearity is applied in the last layer of the output of the generator to produce a probability that corresponds with the one-hot representation of the training data.

During training, the output of the softmax layers becomes the input to the critic network without any rounding step.

To generate a new sample, an argmax operation over each generator's softmax output vectors is applied to produce a valid discrete sample.

The usage of continuous sample distance in the standard Wasserstein GAN for discrete problems as described above creates some discrepancies in the model.

These discrepancies are illustrated in TAB0 .

In TAB0 , we have two different outputs from the generator's softmax with the same real sample reference.

Although the first softmax output produces the same value as the real sample when it is rounded using argmax (hence has discrete distance 0 to the real sample), it has a larger continuous distance compared to the second softmax output which produces one mistake when rounded (has discrete distance 1 to the real sample).

In the discrete case, with a large number of classes, as shown in Table 2 , even though the generator output produces a discrete sample with the same value as the real sample when rounded, there still exists a very large continuous distance.

This difference between continuous and discrete distance becomes greater for a larger number of discrete classes.

Motivated to correct modeling discrepancies as described in Section 2, which occur due to the mismatched use of the standard Wasserstein GAN in discrete problems, we propose a new GAN architecture that is directly based on the Wasserstein distance between two discrete distributions.

Let a vector x = (x(1), x(2), . . .) be a discrete multivariate random variable where each component x(i) can take discrete values from {1, 2, 3, . . . , k}. Let P r and P s be two probability distributions over the set of values for x. The Wasserstein distance between two probability distributions P r and P s is defined as: DISPLAYFORM0 The notation Π(P r , P s ) denotes the set of all joint probability distributions γ(x, x ) whose marginals are P r and P s respectively, and d(x i , x j ) denotes the elementary distance between two samples x i and x j .

We are particularly interested with the sample distance that is defined as the hamming distance (the sum of zero-one distance of each component), i.e: TAB1 shows an example of the sample distance metric.

DISPLAYFORM1 Visible in the formulation above, computing the Wasserstein distance between two discrete probability distributions is a Linear Program (LP) problem for which the runtime is polynomial with respect to the size of problem.

However, for generating real-world discrete distributions, the size of problem grows exponentially.

For example, if the number of variables in vector x is 100, and each variable can take values in the set {1, 2, . . .

, 10} so that k = 10, the size of the LP problem is O(10 100 ) reflecting the number of configurations for x. The resulting LP is intractable to solve.

We follow a similar approach as in by considering the dual formulation of Wasserstein distance.

Kantorovich duality BID2 BID13 tells us that the dual linear program of the Wasserstein distance can computed as: max DISPLAYFORM2 DISPLAYFORM3 The function f maps a sample to a real value.

Note that unlike for the continuous Wasserstein distance, in which the maximization is over all 1-Lipschitz functions without additional constraints, the maximization above is over all functions that satisfy the inequality constraints in Eq. 5.

The dual formulation of the Wasserstein distance is still intractable since the maximization is over all functions that satisfy the inequality constraints.

We aim to approximate the dual Wasserstein distance formulation by replacing f with a family of parameterized functions f w that satisfy the inequality constraints.

The parameterized functions f w are modeled using a neural network.

Unfortunately, it is difficult to construct a neural network architecture to model f w while also explicitly satisfying the inequality constraints involving the discrete sample distance defined in Eq. 3.To overcome the problem of approximating f with neural networks, we note that the maximization in the dual formulation is equivalent to the following optimization: DISPLAYFORM0 subject to: DISPLAYFORM1 where DISPLAYFORM2 Instead of approximating f (x), we aim to design a neural network architecture that approximates h(x, x ) and satisfies the inequality constraints in Eq. 5.

The key idea is that this new optimization is equivalent to the original dual formulation of the Wasserstein distance (explained in the sequel), even though the optimal form for h is not explicitly specified.

Our selected architecture for the generator network employs the same softmax nonlinearity trick for the standard Wasserstein GAN described in Section 2.

The generator network is a parameterized function g θ that maps random noise z to a sample in one-hot representation.

The last layer of the generator network utilizes softmax nonlinearity to produce a probability which corresponds with the one-hot representation of the real sample.

Our key modeling difference lies in the critic network.

The critic network takes two inputs, one from the real samples, and one from the output of the generator.

The architecture of the critic network is visualized in FIG2 .Let y ∈ [0, 1] m×k be the one-hot representation of x where m is the number of variables and k is the number of classes for each variable.

The critic network takes two inputs: y from the real training data, and y from the output of the generator network.

Let us define ρ w as a parameterized function that takes input (y, y ) ∈ [0, 1] 2×m×k and produces an output vector v ∈ [−1, 1] m .

From the generator output y , we compute the rounded samplex .

Let u ∈ {0, 1} m be a vector that contains the element-wise zero one distance between a real training sample x and rounded samplex from the generator, i.e. u(i) = I(x(i) =x (i)).

We define our approximation to the function h as a parameterized function h w that is defined as h w = u T v = u T ρ w (y, y ).

The "filter" vector u ensures that the output of h w always satisfies the inequality constraints DISPLAYFORM3 5.

An illustration of this neural network architecture and construction is provided in FIG2 .As we can see from FIG2 , the critic network consists of two separate sub-networks.

The first sub-network takes input from a batch of samples of the training data, while the second sub-network takes input from a batch of samples produced by the generator.

Each sub-network has its own set of intermediate layers.

The outputs of the first and second layers are concatenated and taken as an input to a fully connected layer which produces a tensor of size n × m. The dimension n indicates the number of samples in a batch, and m is the number of variables.

To produce a tensor v whose values range from -1 to 1, a tanh nonlinearity is applied.

The "filter" tensor u is applied to v via an element-wise multiplication.

The output of the critic network is calculated by taking the sum of the result of the element-wise multiplication of u and v, yielding a vector of n elements containing the value of h w (y, y ) for each pair of real and generated samples.

We also included additional modifications based on theory to facilitate the training of networks.

Note that since h(x, x ) = f (x) − f (x ), we can attempt to enforce this optimum condition known from theory.

If we flip the inputs to h w we will get the negative of the output; i.e. h w (y , y) = −h w (y, y ).

To model this fact, we randomly swapped the sample from the real training data and generator output so that some of the real data was fed to the first sub-network and some to the second sub-network.

If a pair of samples was flipped, we multiplied the output of the network with −1.

Another modification that we applied to the network was to introduce a scaling factor to the softmax function such that the output of the scaled softmax was closer to zero or one.

Specifically, we applied the function: softmax(x)(i) = exp(k·x(i)) j exp(k·x(j)) , for some constant k ≥ 1.

The training algorithm for our proposed architecture is described in Algorithm 1.

Algorithm 1 Discrete Wasserstein GAN 1:

Input: learning rate α, batch size n, the number of critic iteration per generator iteration n critic 2: repeat 3: DISPLAYFORM4 Sample a batch from real data DISPLAYFORM5 Sample a batch of random noise DISPLAYFORM6 end for 8:Sample a batch from real data DISPLAYFORM7 Sample a batch of random noise DISPLAYFORM8 10: DISPLAYFORM9

In contrast with the continuous GANs where many models have been proposed to improve the performance of GAN training, only a few GAN formulations have been proposed for modeling discrete probability distributions.

BID4 use the standard continuous Wassersten GAN with adjustments described in Section 2.

Similar techniques are used by BID12 to address several natural language generation tasks.

augment the original GAN architecture with a maximum likelihood technique and combine the discriminator output with importance sampling from the maximum likelihood training.

propose a Boundaryseeking GAN (BGAN) that trains the generator to produce samples that lie in the decision boundary of the discriminator.

BGAN can be applied for discrete cases provided that the generator outputs a parametric conditional distribution.

Other GAN models BID15 exploit the REINFORCE policy gradient algorithm BID14 to overcome the difficulty of backpropagation in the discrete setting.

BID6 combine adversarial training with Variational Autoencoders BID7 to model discrete probability distributions.

Evaluating the performance of generative models objectively and effectively is hard, since it is difficult to automatically tell whether a generated sample is a valid sample from the real distribution.

Previous research advocates user studies with human graders, especially in image generation tasks, or proxy measures like perplexity and corpus-level BLEU in natural language generation.

However, such techniques are far from ideal to objectively evaluate the performance of GAN models.

To address the limitations above, we propose a synthetic experiment that captures the complexity of modeling discrete distributions, but still has a simple strategy to objectively evaluate performance.

The synthetic experiment is based on a classic tic-tac-toe game.

We generalize the classic 2 player tic-tac-toe game to include arbitrary k players and arbitrary m-by-m board sizes (rather than the default 3-by-3 board).

The goal is to model the true generating distribution P r which is the uniform distribution over valid configurations of the board when a generalized tic-tac-toe game has ended (e.g. the final game state).

We generalized the concept of a valid board in 3-by-3 games, in which one player has a winning state and marks filling a full column, row, or diagonal.

For the purpose of our experiment, we made a simplification to the valid rule, i.e. as long as the board has at least one full column, row and diagonal taken by at least one player, it is considered to be a valid configuration.

FIG3 shows examples of valid and non-valid board configurations.

In our construction above, it is easy to check if a generated sample is a valid sample under the real distribution.

Hence it is possible to validate objectively the performance of a generative model.

Furthermore, it is also easy to sample from the real distribution to create synthetic training data.

We uniformly sample random board configurations, accepting a sample if it is valid, and rejecting it if invalid.

We construct several metrics to track the performance of the model.

The first measure is the percentage of valid samples which characterizes the quality of the samples generated by the generator network.

For a bigger board the percentage of valid samples does not tell much about the progress of learning since it takes a while to get to a valid sample.

We construct another metric which is the average of maximum player's gain.

The maximum player's gain for a board configuration is defined as the maximum number of cells taken by a player in a full column, row, or diagonal.

FIG4 shows the value of maximum player's gain for three different 5-by-5 board configurations.

In the left board, player 2 and 4 have the maximum (3 cells); in the middle board player 2 takes 4 cells; and in the right board, player 2 achieves the maximum of 5 cells.

Note that for k-by-k boards, if the average of maximum player's gain is equal to k, it means that all the samples are valid.

Therefore, closer average of maximum player's gain to k indicates a better quality of samples.

Besides those two metrics, we also track the percentage of unique samples and the percentage of new samples, i.e. samples that do not appear in the training data.

In the experiment, we compare our Discrete Wasserstein GAN model with the standard Wasserstein GAN model (with tricks described in Section 2) on 3-by-3 and 5-by-5 board with 2 players and 8 players.

Note that the number of classes is equal to the number of players plus one since we need an additional class for encoding empty cells.

We restrict the generator and critic networks in both models to have a single hidden layer within fully connected networks to ease training.

As we can see from Figure 4 , our DWGAN networks achieve good performance (in terms of the average of the percentage of valid samples and the maximum player's gain metrics) much faster than the standard WGAN with softmax and one-hot representation tricks.

In both 3-by-3 boards with 2 players and 5-by-5 boards with 8 players our DWGAN networks only take less than a third of the iterations taken by the standard WGAN to achieve similar performance.

We observe that our DWGAN networks have a mode collapse problem that occurs after achieving top performances.

Figure 5a shows that the DWGAN can achieve the average of maximum player's gain close to 5 for a 5-by-5 board in 500 iterations while maintaining the percentage of unique samples close to 100%.

After it produces those diverse samples, the network model begins to suffer from a mode collapse and the percentage of unique samples decrease to less than 10% after iteration 550.

Based on our analysis, this behavior is caused by the fact that the network optimizes the function difference DISPLAYFORM0 , which tends to cause an advantage if the values of g θ (z i ) are not diverse.

To overcome this issue, we add a norm penalty to the critic network optimization, i.e: DISPLAYFORM1 where λ is the penalty constant.

Figure 5b shows the effect of the norm penalty to the performance of DWGAN and its sample diversity.

We observe that the DWGAN network with a norm penalty can achieve 96% valid samples while maintaining the diversity in samples it generates (around 50% unique samples).

To model more complex discrete distributions, we used MNIST digits discretized to binary values BID8 as the training data with the goal to generate new digits with our proposed Discrete Wasserstein GAN.

As a baseline, we trained a standard Wasserstein GAN on the continuous digit dataset.

Similar to our synthetic experiments, we restricted the generator and critic networks to have only a single hidden layer within fully connected networks.

Figure 6 shows that our model produces a similar quality of discretized digit images compared to the continuous value digits produced by the standard Wasserstein GAN trained on continuous-valued data.

We further generated 100 samples from our DWGAN model, prior to mode collapse, illustrating the diversity of samples.

We proposed the Discrete Wasserstein GAN (DWGAN) which approximates the Wasserstein distance between two discrete distributions.

We derived a novel training algorithm and corresponding network architecture for a dual formulation to the problem, and presented promising experimental results.

Our future work focuses on exploring techniques to improve the stability of the training process, and applying our model to other datasets such as for natural language processing.

A linear program (LP) is a convex optimization problem in which the objective and constraint functions are linear.

Consider a vector variable x ∈ R n , matrix A ∈ R n×m , and vectors c ∈ R n , and b ∈ R m .

An LP is given in the following standard form, DISPLAYFORM0 The Lagrange dual function is given by, DISPLAYFORM1 The Lagrange dual problem is to maximize g(λ, ν) subject to λ 0.

Equivalently, the dual problem may be written as an LP in inequality form with vector variable ν ∈ R m , DISPLAYFORM2 The dual of the above problem is equivalent to the original LP in standard form.

Due to the weaker form of Slater's condition, strong duality holds for any LP in standard or inequality form provided that the primal problem is feasible.

Similarly, strong duality holds for LPs if the dual is feasible.

Consider discrete probability distributions over a finite set X with cardinality |X |.

Assume an elementary sample distance d X (x 1 , x 2 ) for x 1 , x 2 ∈ X .

The sample distance evaluates the semantic similarity between members of set X .

Define P r (x) and P s (x) for x ∈ X as two discrete probability distributions.

In this case, we may define the exact discrete Wasserstein distance between P r and P s as a linear program as follows, with D ∈ R |X |×|X | + whose matrix entries correspond to the sample distance DISPLAYFORM0 The dual LP is given as follows.

DISPLAYFORM1 At the optimum it is known that ν = −µ, and the dual LP is equivalent to the following optimization problem.

Note that there still exist |X | × |X | constraints.

DISPLAYFORM2 Example 1.

The following example provides a closer look at the dual optimization problem.

Consider a finite set X = {1, 2, 3}. Let P s (x) be given by the discrete distribution P s (1) = 0.2, P s (2) = 0.7 and P s (3) = 0.1.

Similarly, let P r (x) be given by the discrete distribution P r (1) = 0.4, P r (2) = 0.4, P r (3) = 0.2.

Define the elementary sample distance d X (x 1 , x 2 ) = 1 if x 1 = x 2 and d X (x 1 , x 2 ) = 0 if x 1 = x 2 .

Therefore, the sample distance matrix D for this discrete example is the following: DISPLAYFORM3 The optimal value of the matrix T provides the optimal transport of mass from P s to P r , The objective value of the primal and dual is equal to 0.3 which is the total mass moved from P s to P r .

In the solution to the dual problem, ν = [ 0 −1 0 ] T and µ = −ν.

In this example, it is seen that the optimal ν = −µ. DISPLAYFORM4

For the synthetic experiments, we use Julia v0.5.2 programming language with Knet deep learning framework.

Below is the code containing functions needed to generate tic-tac-toe board.

# np = number of player for j = 1:c_iter # fake samples z = KnetArray(randn(Float32, nz, n)) fv = netG(wG, z, np, softmax_scaling) # real + fake outputC = netC(wC, rv, fv, np, lambda) tl.log_value("output_C", outputC, itC) tl.log_value("output", outputC, it) gC = netC_grad(wC, rv, fv, np, lambda) tl.log_value("grad_C_mean", mean(map(x -> mean(Array(x)), gC)), itC) tl.log_value("grad_C_std", mean(map(x -> std(Array(x)), gC)), itC) for i in 1:length(wC) wC[i] += lrC * gC[i] end it += 1 itC += 1 end if itC % decayitC == 0 lrC = decayC * lrC end # train generator for j = 1:g_iter z = KnetArray(randn(Float32, nz, n)) outputG = netGC (wG, wC, z, rv, np, softmax_scaling, lambda) tl.log_value("output_G", outputG, itG) tl.log_value("output", outputG, it) gG = netGC_grad(wG, wC, z, rv, np, softmax_scaling, lambda) tl.log_value("grad_G_mean", mean(map(x -> mean(Array(x)), gG)), itG) tl.log_value("grad_G_std", mean(map(x -> std(Array(x)), gG)), itG) for i in 1:length ( DISPLAYFORM0

@highlight

We propose a Discrete Wasserstein GAN (DWGAN) model which is based on a dual formulation of the Wasserstein distance between two discrete distributions.