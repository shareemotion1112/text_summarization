We present an end-to-end trained memory system that quickly adapts to new data and generates samples like them.

Inspired by Kanerva's sparse distributed memory, it has a robust  distributed reading and writing mechanism.

The memory is analytically tractable, which enables optimal on-line compression via a Bayesian update-rule.

We formulate it as a hierarchical conditional generative model, where memory provides a rich data-dependent prior distribution.

Consequently, the top-down memory and bottom-up perception are combined to produce the code representing an observation.

Empirically, we demonstrate that the adaptive memory significantly improves generative models trained on both the Omniglot and CIFAR datasets.

Compared with the Differentiable Neural Computer (DNC) and its variants, our memory model has greater capacity and is significantly easier to train.

Recent work in machine learning has examined a variety of novel ways to augment neural networks with fast memory stores.

However, the basic problem of how to most efficiently use memory remains an open question.

For instance, the slot-based external memory in models like Differentiable Neural Computers (DNCs BID10 ) often collapses reading and writing into single slots, even though the neural network controller can in principle learn more distributed strategies.

As as result, information is not shared across memory slots, and additional slots have to be recruited for new inputs, even if they are redundant with existing memories.

Similarly, Matching Networks BID25 BID4 and the Neural Episodic Controller BID21 directly store embeddings of data.

They therefore require the volume of memory to increase with the number of samples stored.

In contrast, the Neural Statistician BID7 summarises a dataset by averaging over their embeddings.

The resulting "statistics" are conveniently small, but a large amount of information may be dropped by the averaging process, which is at odds with the desire to have large memories that can capture details of past experience.

Historically developed associative memory architectures provide insight into how to design efficient memory structures that store data in overlapping representations.

For example, the Hopfield Net BID14 pioneered the idea of storing patterns in low-energy states in a dynamic system.

This type of model is robust, but its capacity is limited by the number of recurrent connections, which is in turn constrained by the dimensionality of the input patterns.

The Boltzmann Machine BID1 lifts this constraint by introducing latent variables, but at the cost of requiring slow reading and writing mechanisms (i.e. via Gibbs sampling).

This issue is resolved by Kanerva's sparse distributed memory model BID15 , which affords fast reads and writes and dissociates capacity from the dimensionality of input by introducing addressing into a distributed memory store whose size is independent of the dimension of the data 1 .In this paper, we present a conditional generative memory model inspired by Kanerva's sparse distributed memory.

We generalise Kanerva's original model through learnable addresses and reparametrised latent variables BID23 BID17 BID5 .

We solve the challenging problem of learning an effective memory writing operation by exploiting the analytic tractability of our memory model -we derive a Bayesian memory update rule that optimally trades-off preserving old content and storing new content.

The resulting hierarchical generative model has a memory dependent prior that quickly adapts to new data, providing top-down knowledge in addition to bottom-up perception from the encoder to form the latent code representing data.

As a generative model, our proposal provides a novel way of enriching the often over-simplified priors in VAE-like models BID22 ) through a adaptive memory.

As a memory system, our proposal offers an effective way to learn online distributed writing which provides effective compression and storage of complex data.

Our memory architecture can be viewed as an extension of the variational autoencoder (VAE) BID23 BID17 , where the prior is derived from an adaptive memory store.

A VAE has an observable variable x and a latent variable z. Its generative model is specified by a prior distribution p θ (z) and the conditional distribution p θ (x|z).

The intractable posterior p θ (z|x) is approximated by a parameterised inference model q φ (z|x).

Throughout this paper, we use θ to represent the generative model's parameters, and φ to represent the inference model's parameters.

All parameterised distributions are implemented as multivariate Gaussian distributions with diagonal covariance matrices, whose means and variances are outputs from neural networks as in BID23 BID17 .We assume a dataset with independently and identically distributed (iid) samples D = {x 1 , . . .

, x n , . . . , x N }.

The objective of training a VAE is to maximise its log-likelihood DISPLAYFORM0 .

This can be achieved by jointly optimising θ and φ for a variational lower-bound of the likelihood (omitting the expectation over all x for simplicity): DISPLAYFORM1 where the first term can be interpreted as the negative reconstruction loss for reconstructing x using its approximated posterior sample from q φ (z|x), and the second term as a regulariser that encourages the approximated posterior to be near the prior of z.

To introduce our model, we use the concept of an exchangeable episode: X = {x 1 , . . .

, x t , . . . , x T } ⊂ D is a subset of the entire dataset whose order does not matter.

The objective of training is the expected conditional log-likelihood BID5 , DISPLAYFORM0 The equality utilises the conditional independence of x t given the memory M , which is equivalent to the assumption of an exchangeable episode X BID2 .

We factorise the joint distribution of p(X, M ) into the marginal distribution p(X) and the posterior p(M |X), so that computing p(M |X) can be naturally interpreted as writing X into the memory.

We propose this scenario as a general and principled way of formulating memory-based generative models, since J is directly related to the mutual information DISPLAYFORM1 As the entropy of the data H(X) is a constant, maximising J is equivalent to maximising I(X; M ), the mutual information between the memory and the episode to store.

We write the collection of latent variables corresponding to the observed episode X as Y = {y 1 , . . .

, y t , . . .

, y T } and Z = {z 1 , . . . , z t , . . . , z T }.

As illustrated in FIG0 , the joint distribution of the generative model can be factorised as DISPLAYFORM0 The first equality uses the conditional independence of z t , y t , x t given M , shown by the "plates" in FIG0 .

The memory M is a K × C random matrix with the matrix variate Gaussian distribution BID11 : where R is a K × C matrix as the mean of M , U is a K × K matrix that provides the covariance between rows of M , and V is a C × C matrix providing covariances between columns of M .

This distribution is equivalent to the multivariate Gaussian distribution of vectorised M : DISPLAYFORM1 DISPLAYFORM2 , where vec (·) is the vectorisation operator and ⊗ denotes the Kronecker product.

We assume independence between the columns but not the rows of M , by fixing V to be the identity matrix I C and allow the full degree of freedom for U .

Since our experiments suggest the covariance between rows is useful for coordinating memory access, this setting balances simplicity and performance FIG0 )

.Accompanying M are the addresses A, a K × S real-value matrix that is randomly initialised and is optimised through back-propagation.

To avoid degeneracy, rows of A are normalised to have L2-norms of 1.

The addressing variable y t is used to compute the weights controlling memory access.

As in VAEs, the prior p θ (y t ) is an isotropic Gaussian distribution N (0, 1).

A learned projection b t = f (y t ) then transforms y t into a S × 1 key vector.

The K × 1 vector w t , as weights across the rows of M , is computed via the product: DISPLAYFORM3 The projection f is implemented as a multi-layer perception (MLP), which transforms the distribution of y t , as well as w t , to potentially non-Gaussian distributions that may better suit addressing.

The code z t is a learned representation that generates samples of x t through the parametrised conditional distribution p θ (x t |z t ).

This distribution is tied for all t ∈ {1 . . .

T }.

Importantly, instead of the isotropic Gaussian prior, z t has a memory dependent prior: DISPLAYFORM4 whose mean is a linear combination of memory rows, with the noise covariance matrix fixed as an identity matrix by setting σ 2 = 1.

This prior results in a much richer marginal distribution, because of its dependence on memory and the addressing variable DISPLAYFORM5 In our hierarchical model, M is a global latent variable for an episode that captures statistics of the entire episode BID4 BID7 , while the local latent variables y t and z t capture local statistics for data x t within an episode.

To generate an episode of length T , we first sample M once, then sample y t , z t , and x t sequentially for each of the T samples.3.2 THE READING INFERENCE MODEL As illustrated in FIG0 , the approximated posterior distribution is factorised using the conditional independence: DISPLAYFORM6 where q φ (y t |x t ) is a parameterised approximate posterior distribution.

The posterior distribution q φ (z t |x t , y t , M ) refines the (conditional) prior distribution p θ (z t |y t , M ) with additional evidence from x t .

This parameterised posterior takes the concatenation of x t and the mean of p θ (z t |y t , M ) (eq. 6) as input.

The constant variance of p θ (z t |y t , M ) is omitted.

Similar to the generative model, q φ (y t |x t ) is shared for all t ∈ {1 . . .

T }.

A central difficulty in updating memory is the trade-off between preserving old information and writing new information.

It is well known that this trade-off can be balanced optimally through Bayes' rule MacKay (2003) .

From the generative model perspective (eq. 2), it is natural to interpret memory writing as inference -computing the posterior distribution of memory p(M |X).

This section considers both batch inference -directly computing p(M |X) and on-line inference -sequentially accumulating evidence from x 1 , . . .

, x T .Following FIG0 , the approximated posterior distribution of memory can be written as DISPLAYFORM0 The last line uses one sample of y t , x t to approximate the intractable integral.

The posterior of the addressing variable q φ (y t |x t ) is the same as in section 3.2, and the posterior of code q φ (z t |x t ) is a parameterised distribution.

We use the short-hand DISPLAYFORM1 when Y, Z are sampled as described here.

We abuse notation in this section and use Z = (z 1 ; , . . . , ; z T ) as a T ×C matrix with all the observations in an episode, and W = (w 1 ; . . . ; w T ) as a T × K matrix with all corresponding weights for addressing.

Given the linear Gaussian model (eq. 6), the posterior of memory p θ (M |Y, Z) is analytically tractable, and its parameters R and U can be updated as follows: DISPLAYFORM2 where ∆ is the prediction error before updating the memory, Σ c is a T × K matrix providing the cross-covariance between Z and M , Σ ξ is a T × T diagonal matrix whose diagonal elements are the noise variance σ 2 and Σ z is a T × T matrix that encodes the covariance for z 1 , . . .

, z T .

This update rule is derived from applying Bayes' rule to the linear Gaussian model (Appendix E).

The prior parameters of p(M ), R 0 and U 0 are trained through back-propagation.

Therefore, the prior of M can learn the general structure of the entire dataset, while the posterior is left to adapt to features presented in a subset of data observed within a given episode.

The main cost of the update rule comes from inverting Σ z , which has a complexity of O(T 3 ).

One may reduce the per-step cost via on-line updating, by performing the update rule using one sample at a time -when X = x t , Σ z is a scalar which can be inverted trivially.

According to Bayes' rule, updating using the entire episode at once is equivalent to performing the one-sample/on-line update iteratively for all observations in the episode.

Similarly, one can perform intermediate updates using mini-batch with size between 1 and T .Another major cost in the update rule is the storage and multiplication of the memory's row-covariance matrix U , with the complexity of O(K 2 ).

Although restricting this covariance to diagonal can reduce this cost to O(K), our experiments suggested this covariance is useful for coordinating memory accessing FIG0 .

Moreover, the cost of O(K 2 ) is usually small, since parameters of the model are dominated by the encoder and decoder.

Nevertheless, a future direction is to investigating low-rank approximation of U that better balance cost and performance.3.4 TRAINING To train this model, we optimise a variational lower-bound of the conditional likelihood J (eq. 2), which can be derived in a fashion similar to standard VAEs: DISPLAYFORM3 To maximise this lower bound, we sample y t , z t from q φ (y t , z t |x t , M ) to approximate the inner expectation.

For computational efficiency, we use a mean-field approximation for the memoryusing the mean R in the place of memory samples (since directly sampling M requires expensive Cholesky decomposition of the non-diagonal matrix U ).

Alternatively, we can further exploit the analytical tractability of the Gaussian distribution to obtain distribution-based reading and writing operations (Appendix F).Inside the bracket, the first term is the usual VAE reconstruction error.

The first KL-divergence penalises complex addresses, and the second term penalises deviation of the code z t from the memory-based prior.

In this way, the memory learns useful representations that do not rely on complex addresses, and the bottom-up evidence only corrects top-down memory reading when necessary.3.5 ITERATIVE SAMPLING An important feature of Kanerva's sparse distributed memory is its iterative reading mechanism, by which output from the model is fed back as input for several iterations.

Kanerva proved that the dynamics of iterative reading will decrease errors when the initial error is within a generous range, converging to a stored memory BID15 .

A similar iterative process is also available in our model, by repeatedly feeding-back the reconstructionx t .

This Gibbs-like sampling follows the loop in FIG0 .

While we cannot prove convergence, in our experiments iterative reading reliably improves denoising and sampling.

To understand this process, notice that knowledge about memory is helpful in reading, which suggests using q φ (y t |x t , M ) instead of q φ (y t |x t ) for addressing (section 3.2).

Unfortunately, training a parameterised model with the whole matrix M as input can be prohibitively costly.

Nevertheless, it is well-known in the coding literature that such intractable posteriors that usually arise in non-tree graphs (as in FIG0 ) can be approximated efficiently by loopy belief-propagation, as has been used in algorithms like Turbo coding BID8 .

Similarly, we believe iterative reading works in our model because q φ (y t |x t ) models the local coupling between x t and y t well enough, so iterative sampling with the rest of the model is likely to converge to the true posterior q φ (y t |x t , M ).

Future research will seek to better understand this process.

Details of our model implementation are described in Appendix C. We use straightforward encoder and decoder models in order to focus on evaluating the improvements provided by an adaptive memory.

In particular, we use the same model architecture for all experiments with both Omniglot and CIFAR dataset, changing only the the number of filters in the convolutional layers, memory size, and code size.

We always use the on-line version of the update rule (section 3.3).

The Adam optimiser was used for all training and required minimal tuning for our model BID16 .

In all experiments, we report the value of variational lower bound (eq. 12) L divided by the length of episode T , so the per-sample value can be compared with the likelihood from existing models.

We first used the Omniglot dataset to test our model.

This dataset contains images of hand-written characters with 1623 different classes and 20 examples in each class BID18 .

This large variation creates challenges for models trying to capture the entire complex distribution.

We use a 64 × 100 memory M , and a smaller 64 × 50 address matrix A. For simplicity, we always randomly sample 32 images from the entire training set to form an "episode", and ignore the class labels.

This represents a worst case scenario since the images in an episode will tend to have relatively little redundant information for compression.

We use a mini-batch size of 16, and optimise the variational lower-bound (eq. 12) using Adam with learning rate 1 × 10 −4 .We also tested our model with the CIFAR dataset, in which each 32 × 32 × 3 real-valued colour image contains much more information than a binary omniglot pattern.

Again, we discard all the label information and test our model in the unsupervised setting.

To accommodate the increased complexity of CIFAR, we use convolutional coders with 32 features at each layer, use a code size of 200, and a 128 × 200 memory with 128 × 50 address matrix.

All other settings are identical to experiments with Omniglot.

We first use the 28 × 28 binary Omniglot from BID6 and follow the same split of 24,345 training and 8,070 test examples.

We first compare the training process of our model with a baseline VAE model using the exact same encoder and decoder.

Note that there is only a modest increase of parameters in the Kanerva Machine compared the VAE since the encoder and decoder dominates the model parameters.

Figure 2: The negative variational lower bound (left), reconstruction loss (central), and KL-Divergence (right) during learning.

The dip in the KL-divergence suggests that our model has learned to use the memory.

Fig. 2 shows learning curves for our model along with those for the VAE trained on the Omniglot dataset.

We plot 4 randomly initialised instances for each model.

The training is stable and insensitive to initialisation.

Fig. 2 (left) shows that our model reached a significantly lower negative variational lower-bound versus the VAE.

Fig. 2 (central) and (right) further shows that the Kanerva Machine achieved better reconstruction and KL-divergence.

In particular, the KL-divergence of our model "dips" sharply from about the 2000th step, implying our model learned to use the memory to induce a more informative prior.

FIG0 confirms this: the KL-divergence for z t has collapsed to near zero, showing that the top-down prior from memory q φ (z t |y t , M ) provides most of the information for the code.

This rich prior is achieved at the cost of an additional KL-divergence for y t FIG0 , right) which is still much lower than the KL-divergence for z t in a VAE.

Similar training curves are observed for CIFAR training FIG0 .

BID9 also observed such KL-divergence dips with a memory model.

They report that the reduction in KL-divergence, rather than the reduction in reconstruction loss, was particularly important for improving sample quality, which we also observed in our experiments with Omniglot and CIFAR.At the end of training, our VAE reached a negative log-likelihood (NLL) of ≤ 112.7 (the lower-bound of likelihood), which is worse than the state-of-the-art unconditioned generation that is achieved by rolling out 80 steps of a DRAW model (NLL of 95.5, BID22 , but comparable to results with IWAE training (NLL of 103.4, BID6 .

In contrast, with the same encoder and decoders, the Kanerva Machine achieve conditional NLL of 68.3.

It is not fair to directly compare our results with unconditional generative models since our model has the advantage of its memory contents.

Nevertheless, the dramatic improvement of NLL demonstrates the power of incorporating an adaptive memory into generative models.

Fig. 3 (left) shows examples of reconstruction at the end of training; as a signature of our model, the weights were well distributed over the memory, illustrating that patterns written into the memory were superimposed on others.iterations Figure 3 : Left: reconstruction of inputs and the weights used in reconstruction, where each bin represents the weight over one memory slot.

Weights are widely distributed across memory slots.

Right: denoising through iterative reading.

In each panel: the first column shows the original pattern, the second column (in boxes) shows the corrupted pattern, and the following columns show the reconstruction after 1, 2 and 3 iterations.

We generalise "one-shot" generation from a single image BID22 , or a few sample images from a limited set of classes BID7 BID4 , to a batch of images with many classes and samples.

To better illustrate how samples are shaped by the conditioning data, in this section we use the same trained models, but test them using episodes with samples from only 2, 4 or 12 classes (omniglot characters) 2 .

FIG1 compares samples from the VAE and the Kanerva Machine.

While initial samples from our model (left most columns) are visually about as good as those from the VAE, the sample quality improved in consecutive iterations and the final samples clearly reflects the statistics of the conditioning patterns.

Most samples did not change much after the 6th iteration, suggesting the iterative sampling had converged.

Similar conditional samples from CIFAR are shown in Fig. 5 .

Notice that this approach, however, does not apply to VAEs, since VAEs do not have the structure we discussed in section 3.5.

This is illustrated in FIG3 by feeding back output from VAEs as input to the next iteration, which shows the sample quality did not improve after iterations.

Figure 5: Comparison of samples from CIFAR.

The 24 conditioning images (top-right) are randomly sampled from the entire CIFAR dataset, so they contains a mix of many classes.

Samples from the matched VAE are blurred and lack meaningful local structure.

On the other hand, samples from the Kanerva Machine have clear local structures, despite using the same encoder and decoder as the VAE.

The 5 columns show samples after 0, 2, 4, 6, and 8 iterations.

To further examine generalisation, we input images corrupted by randomly positioned 12 × 12 blocks, and tested whether our model can recover the original image through iterative reading.

Our model was not trained on this task, but Fig. 3 (right) shows that, over several iterations, input images can be recovered.

Due to high ambiguity, some cases (e.g., the second and last) ended up producing incorrect but still reasonable patterns.

The structure of our model affords interpretability of internal representations in memory.

Since representations of data x are obtained from a linear combination of memory slots (eq. 6), we expect linear interpolations between address weights to be meaningful.

We examined interpolations by computing 2 weight vectors from two random input images, and then linearly interpolating between these two vectors.

These vectors were then used to read z t from memory (eq. 6), which is then decoded to produce the interpolated images.

FIG2 in Appendix A shows that interpolating between these access weights indeed produces meaningful and smoothly changing images.

Figure 6 : Left: the training curves of DNC and Kanerva machine both shows 6 instances with the best hyperparameter configuration for each model found via grid search.

DNCs were more sensitive to random initilisation, slower, and plateaued with larger error.

Right: the test variational lower-bounds of a DNC (dashed lines) and a Kanerva Machine as a function of different episode sizes and different sample classes.

This section compares our model with the Differentiable Neural Computer (DNC, BID10 , and a variant of it, the Least Recently Used Architecture (LRUA, BID24 .

We test these using the same episode storage and retrieval task as in previous experiments with Omniglot data.

For a fair comparison, we fit the DNC models into the same framework, as detailed in Appendix D. Fig. 6 (left) illustrates the process of training the DNC and the Kanerva Machine.

The LRUA did not passed the loss level of 150, so we did not include it in the figure.

The DNC reached a test loss close to 100, but was very sensitive to hyper-parameters and random initialisation: only 2 out of 6 instances with the best hyper-parameter configuration (batch size = 16, learning rate= 3 × 10 −4 ) found by grid search reached this level.

On the other hand, the Kanerva Machine was robust to these hyper-parameters, and worked well with batch sizes between 8 and 64, and learning rates between 3 × 10 −5 and 3 × 10 −4 .

The Kanerva Machine trained fastest with batch size 16 and learning rate 1 × 10 −4 and eventually converged below 70 test loss with all tested configurations.

Therefore, the Kanerva Machine is significantly easier to train, thanks to principled reading and writing operations that do not depend on any model parameter.

We next analysed the capacity of our model versus the DNC by examining the lower bound of then likelihood when storing and then retrieving patterns from increasingly large episodes.

As above, these models are still trained with episodes containing 32 samples, but are tested on much larger episodes.

We tested our model with episodes containing different numbers of classes and thus varying amounts of redundancy.

Fig. 6 (right) shows both models are able to exploit this redundancy, since episodes with fewer classes (but the same number of images) have lower reconstruction losses.

Overall, the Kanerva Machine generalises well to larger episodes, and maintained a clear advantage over the DNC (as measured by the variational lower-bound).

In this paper, we present the Kanerva Machine, a novel memory model that combines slow-learning neural networks and a fast-adapting linear Gaussian model as memory.

While our architecture is inspired by Kanerva's seminal model, we have removed the assumption of a uniform data distribution by training a generative model that flexibly learns the observed data distribution.

By implementing memory as a generative model, we can retrieve unseen patterns from the memory through sampling.

This phenomenon is consistent with the observation of constructive memory neuroscience experiments BID12 .Probabilistic interpretations of Kanerva's model have been developed in previous works: Anderson (1989) explored a conditional probability interpretation of Kanerva's sparse distributed memory, and generalised binary data to discrete data with more than two values.

BID0 provides an approximate Bayesian interpretation based on importance sampling.

To our knowledge, our model is the first to generalise Kanerva's memory model to continuous, non-uniform data while maintaining an analytic form of Bayesian inference.

Moreover, we demonstrate its potential in modern machine learning through integration with deep neural networks.

Other models have combined memory mechanisms with neural networks in a generative setting.

For example, BID19 used attention to retrieve information from a set of trainable parameters in a memory matrix.

Notably, the memory in this model is not updated following learning.

As a result, the memory does not quickly adapt to new data as in our model, and so is not suited to the kind of episode-based learning explored here.

BID5 used discrete (categorical) random variables to address an external memory, and train the addressing mechanism, together with the rest of the generative model, though a variational objective.

However, the memory in their model is populated by storing images in the form of raw pixels.

Although this provides a mechanism for fast adaptation, the cost of storing raw pixels may be overwhelming for large data sets.

Our model learns to to store information in a compressed form by taking advantage of statistical regularity in the images via the encoder at the perceptual level, the learned addresses, and Bayes' rule for memory updates.

Central to an effective memory model is the efficient updating of memory.

While various approaches to learning such updating mechanisms have been examined recently BID10 BID7 BID24 , we designed our model to employ an exact Bayes' update-rule without compromising the flexibility and expressive power of neural networks.

The compelling performance of our model and its scalable architecture suggests combining classical statistical models and neural networks may be a promising direction for novel memory models in machine learning.

This section reviews Kanerva's sparse distributed memory BID15 .

For consistency with the rest of this paper, many of the notations are different from Kanerva's description.

In contrast to many recent models, Kanerva's memory model is characterised by its distributed reading and writing operations.

The model has two main components: a fixed table of addresses A pointing to a modifiable memory M .

Both A and M have the same size of K × D, where K is the number of addresses that and D is the input dimensionality.

Kanerva assumes all the inputs are uniform random vectors y ∈ {−1, 1} D .

Therefore, the fixed addresses A i are uniformly randomly sampled from {−1, 1} D to reflect the input statistics.

An input y is compared with each address A k in A through the Hamming distance.

For binary vectors a, b ∈ {−1, 1} D , the Hamming distance can be written as h(a, b) = 1 2 (D − a · b) where · represents inner product between two vectors.

An address k is selected when the hamming distance between x and A k is smaller than a threshold τ , so the selection can be summarised by the binary weight vector: DISPLAYFORM0 During writing, a pattern x is stored into M by adding M k ← M k + w k x. For reading, the memory contents pointed to by all the selected addresses are summed together to pass a threshold at 0 to produce a read out:x DISPLAYFORM1 This reading process can be iterated several times by repeatedly feeding-back the outputx as input.

It has been shown analytically by Kanerva that when both K and D are large enough, a small portion of the addresses will always be selected, thus the operations are sparse and distributed.

Although an address' content may be over-written many times, the stored vectors can be retrieved correctly.

Moreover, Kanerva proved that even a significantly corrupted query can be discovered from the memory through iterative reading.

However, the application of Kanerva's model is restricted by the assumption of a uniform and binary data distribution, on which Kanerva's analyses and bounds of performance rely BID15 .

Unfortunately, this assumption is rarely true in practice, since real-world data typically lie on low-dimensional manifolds, and binary representation of data is less efficient in high-level neural network implementations that are heavily optimised for floating-point numbers.

C MODEL DETAILS FIG4 shows the architecture of our model compared with a standard VAE.

For all experiments, we use a convolutional encoder to convert input images into 2C embedding vectors e(x t ), where C is the code size (dimension of z t ).

The convolutional encoder has 3 consecutive blocks, where each block is a convolutional layer with 4 × 4 filter with stride 2, which reduces the input dimension, followed by a basic ResNet block without bottleneck BID13 .

All the convolutional layers have the same number of filters, which is either 16 or 32 depending on the dataset.

The output from the blocks is flattened and linearly projected to a 2C dimensional vector.

The convolutional decoder mirrors this structure with transposed convolutional layers.

All the "MLP" boxes in FIG4 are 2-layer multi-layer perceptron with ReLU non-linearity in between.

We found that adding noise to the input into q φ (y t |x t ) helped stabilise training, possibly by restricting the information in the addresses.

The exact magnitude of the added noise matters little, and we use Gaussian noise with zero mean and standard deviation of 0.2 for all experiments.

We use Bernoulli likelihood function for Omniglot dataset, and Gaussian likelihood function for CIFAR.

To avoid Gaussian likelihood collapsing, we added uniform noise U(0, 1 256 ) to CIFAR images during training.

For a fair comparison, we wrap the differentiable neural computer (DNC) with the same interface as the Kanerva memory so that it can simply replace the memory M in FIG4 .

More specifically, the DNC receives the addressing variable y t with the same size and sampled the same ways as described in the main text in reading and writing stages.

During writing it also receives z t sampled from q φ (z t |x t ) as input, by concatenating y t and z t together as input into the memory controller.

Since DNCs do not have separated reading and writing stages, we separated this two process in our experiments: during writing, we discard the read-out from the DNC, and only keep its state as the memory; during reading, we discard the state at each step so it cannot be used for storing new information.

In addition, we use a 2-layer MLP with 200 hidden neurons and ReLU nonlinearity as the controller instead of the commonly used LSTM to avoid the recurrent state being used as memory and interference with DNC's external memory.

Another issue with off-the-shelf DNC BID10 BID24 is that controllers may generate output bypassing the memory, which can be particularly confusing in our auto-encoding setting by simply ignoring the memory and functioning as a skip connection.

We avoid this situation by removing this controller output and ensure that the DNC only reads-out from its memory.

Further, to focus on the memory performance, we remove Figure 10: Covariance between memory rows is important.

The two curves shows the test loss (negative variational lower bound) as a function of iterations.

Four models using full K × K covariance matrix U are shown by red curves and four models using diagonal covariance matrix are shown in blue.

All other settings for these 8 models are the same (as described in section 4).

These 8 models are trained on machines with similar setup.

The models using full covariance matrices were slightly slower per-iteration, but the test loss decreased far more quickly.the bottom-up stream in our model that compensates for the memory.

This means directly sampling z t from p θ (z t |y t , M ), instead of p θ (z t |x t , y t , M ), for the decoder p θ (x t |z t ), forcing the model to reconstruct solely using read-outs from the memory.

The negative variational lower bound, reconstruction loss, and total KL-divergence during CIFAR training.

Although the difference between the lower bound objective is smaller than that during Omniglot training, the general patterns of these curves are similar to those in Fig. 2 .

The relatively small difference in KL-divergence significantly influences sample quality.

Notice at the time of our submission, the training is continuing and the advantage of the Kanerva Machine over the VAE is increasing.

Eq. 6 defines a linear Gaussian model.

Using notations in the main paper, can write the joint distribution p(vec (Z) , vec(M )) = N (vec (Z) , vec(M ); µ j , Σ j ), where DISPLAYFORM0 DISPLAYFORM1 We can then use the conditional formula for the Gaussian to derive the posterior distribution p(vec (M ) |vec (Z)) = N (vec (M ) ; µ p , Σ p ), using the property Kronecker product: DISPLAYFORM2 From properties of matrix variate Gaussian distribution, the above two equations can be re-arranged to the update rule in eq. 9 to 11.F DISTRIBUTION-BASED READING AND WRITING While the model we described in this paper works well using samples from q φ (z t |x t ) for writing to the memory (section 3.3) and the mean-field approximation during reading (section 3.4), here we describe an alternative that fully exploits the analytic tractability of the Gaussian distribution.

To simplify notation, we use ψ = {R, U, V } for all parameters of the memory.

<|TLDR|>

@highlight

A generative memory model that combines slow-learning neural networks and a fast-adapting linear Gaussian model as memory.