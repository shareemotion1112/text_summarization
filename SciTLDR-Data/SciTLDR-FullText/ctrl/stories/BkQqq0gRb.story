This paper develops variational continual learning (VCL), a simple but general framework for continual learning that fuses online variational inference (VI) and recent advances in Monte Carlo VI for neural networks.

The framework can successfully train both deep discriminative models and deep generative models in complex continual learning settings where existing tasks evolve over time and entirely new tasks emerge.

Experimental results show that VCL outperforms state-of-the-art continual learning methods on a variety of tasks, avoiding catastrophic forgetting in a fully automatic way.

Continual learning (also called life-long learning and incremental learning) is a very general form of online learning in which data continuously arrive in a possibly non i.i.d.

way, tasks may change over time (e.g. new classes may be discovered), and entirely new tasks can emerge BID43 BID47 BID39 .

What is more, continual learning systems must adapt to perform well on the entire set of tasks in an incremental way that avoids revisiting all previous data at each stage.

This is a key problem in machine learning since real world tasks continually evolve over time (e.g. they suffer from covariate and dataset shift) and the size of datasets often prohibits frequent batch updating.

Moreover, practitioners are often interested in solving a set of related tasks that benefit from being handled jointly in order to leverage multi-task transfer.

Continual learning is also of interest to cognitive science, being an intrinsic human ability.

The ubiquity of deep learning means that it is important to develop deep continual learning methods.

However, it is challenging to strike a balance between adapting to recent data and retaining knowledge from old data.

Too much plasticity leads to the infamous catastrophic forgetting problem BID34 BID36 BID13 and too much stability leads to an inability to adapt.

Recently there has been a resurgence of interest in this area.

One approach trains individual models on each task and then carries out a second stage of training to combine them BID28 .

A more elegant and more flexible approach maintains a single model and uses a single type of regularized training that prevents drastic changes in the parameters which have a large influence on prediction, but allows other parameters to change more freely BID29 BID26 BID50 .

The approach developed here follows this venerable work, but is arguably more principled, extensible and automatic.

This paper is built on the observation that there already exists an extremely general framework for continual learning: Bayesian inference.

Critically, Bayesian inference retains a distribution over model parameters that indicates the plausibility of any setting given the observed data.

When new data arrive, we combine what previous data have told us about the model parameters (the previous posterior) with what the current data are telling us (the likelihood).

Multiplying and renormalizing yields the new posterior, from which point we can recurse.

Critically, the previous posterior constrains parameters that strongly influence prediction, preventing them from changing drastically, but it allows other parameters to change.

The wrinkle is that exact Bayesian inference is typically intractable and so approximations are required.

Fortunately, there is an extensive literature on approximate inference for neural networks.

We merge online variational inference (VI) BID11 BID42 BID4 with Monte Carlo VI for neural networks BID3 to yield variational continual learning (VCL).

In addition, we extend VCL to include a small episodic memory by combining VI with the coreset data summarization method BID0 BID19 .

We demonstrate that the framework is general, applicable to both deep discriminative models and deep generative models, and that it yields excellent performance.

Consider a discriminative model that returns a probability distribution over an output y given an input x and parameters θ, that is p(y|θ, x).

Below we consider the specific case of a softmax distribution returned by a neural network with weight and bias parameters, but we keep the development general for now.

In the continual learning setting, the goal is to learn the parameters of the model from a set of sequentially arriving datasets {x DISPLAYFORM0 where, in principle, each might contain a single datum, N t = 1.

Following a Bayesian approach, a prior distribution p(θ) is placed over θ.

The posterior distribution after seeing T datasets is recovered by applying Bayes' rule: DISPLAYFORM1 Here the input dependence has been suppressed on the right hand side to lighten notation.

We have used the shorthand D t = {y DISPLAYFORM2 .

Importantly, a recursion has been identified whereby the posterior after seeing the T -th dataset is produced by taking the posterior after seeing the (T − 1)-th dataset, multiplying by the likelihood and renormalizing.

In other words, online updating emerges naturally from Bayes' rule.

In most cases the posterior distribution is intractable and approximation is required, even when forming the first posterior p(θ|D 1 ) ≈ q 1 (θ) = proj(p(θ)p(D 1 |θ)).

Here q(θ) = proj(p * (θ)) denotes a projection operation that takes the intractable un-normalized distribution p * (θ) and returns a tractable normalized approximation q(θ).

The field of approximate inference provides several choices for the projection operation including i) Laplace's approximation, ii) variational KL minimization, iii) moment matching, and iv) importance sampling.

Having approximated the first posterior distribution, subsequent approximations can be produced recursively by combining the approximate posterior distribution with the likelihood and projecting, that is p(θ|D 1:T ) ≈ q T (θ) = proj(q T −1 (θ)p(D T |θ)).

In this way online updating is supported.

This general approach leads, for the four projection operators previously identified, to i) Laplace propagation BID46 , ii) online VI BID11 BID42 ) also known as streaming variational Bayes BID4 , iii) assumed density filtering BID33 and iv) sequential Monte Carlo BID30 .

In this paper the online VI approach is used as it typically outperforms the other methods for complex models in the static setting ) and yet it has not been applied to continual learning of neural networks.

Variational continual learning employs a projection operator defined through a KL divergence minimization over the set of allowed approximate posteriors Q, DISPLAYFORM0 The zeroth approximate distribution is defined to be the prior, q 0 (θ) = p(θ).

Z t is the intractable normalizing constant of p * t (θ) = q t−1 (θ) p(D t |θ) and is not required to compute the optimum.

VCL will perform exact Bayesian inference if the true posterior is a member of the approximating family, p(θ|D 1 , D 2 , . . .

, D t ) ∈ Q at every step t. Typically this will not be the case and we might worry that performing repeated approximations may accumulate errors causing the algorithm to forget old tasks, for example.

Furthermore, the minimization at each step may also be approximate (e.g. due to employing an additional Monte Carlo approximation) and so additional information may be lost.

In order to mitigate this potential problem, we extend VCL to include a small representative set of data from previously observed tasks that we call the coreset.

The coreset is analogous to an episodic memory that retains key information (in our case, important training data points) from previous tasks which the algorithm can revisit in order to refresh its memory of them.

The use of an episodic memory for continual learning has also been explored by BID31 .

Input: Prior p(θ).

Output: Variational and predictive distributions at each step {qt(θ), p(y * |x * , D1:t)} T t=1 .

Initialize the coreset and variational approximation: C0 ← ∅,q0 ← p. for t = 1 . . .

T do Observe the next dataset Dt.

Ct ← update the coreset using Ct−1 and Dt.

Update the variational distribution for non-coreset data points: DISPLAYFORM0 Compute the final variational distribution (only used for prediction, and not propagation): DISPLAYFORM1 Perform prediction at test input x * : p(y * |x * , D1:t) = qt(θ)p(y * |θ, x * )dθ. end for Algorithm 1 describes coreset VCL.

For each task, the new coreset C t is produced by selecting new data points from the current task and a selection from the old coreset C t−1 .

Any heuristic can be used to make these selections, e.g. K data points can be selected at random from D t and added to C t−1 to form an unbiased new coreset C t .

Alternatively, the greedy K-center algorithm BID12 can be used to return K data points that are guaranteed to be spread throughout the input space.

Next, a variational recursion is developed.

Bayes' rule can be used to decompose the true posterior taking care to break out contributions from the coreset, DISPLAYFORM2 Here the variational distributionq t (θ) approximates the contribution to the posterior from the noncoreset data points.

A recursion is identified by noting DISPLAYFORM3 Hence propagation is performed viaq t (θ) = proj(q t−1 (θ)p(D t ∪ C t−1 \ C t |θ)) with VCL employing the variational KL projection.

A further projection step is needed before performing prediction q t (θ) = proj(q t (θ)p(C t |θ)).

In this way the coreset is incorporated into the approximate posterior directly before prediction which helps mitigate any residual forgetting.

From a more general perspective, coreset VCL is equivalent to a message-passing implementation of VI in which the coreset data point updates are scheduled after updating the other data.

The VCL framework is general and can be applied to many discriminative probabilistic models.

Here we apply it to continual learning of deep fully-connected neural network classifiers.

Before turning to the application of VCL, we first consider the architecture of neural networks suitable for performing continual learning.

In simple instances of discriminative continual learning, where data are arriving in an i.i.d.

way or where only the input distribution p(x 1:T ) changes over time, a standard single-head discriminative neural network suffices.

In many cases the tasks, although related, might involve different output variables.

Standard practice in multi-task learning BID1 uses networks that share parameters close to the inputs but with separate heads for each output, hence multi-head networks.

Graphical models depicting the network architecture for deep discriminative and deep generative models are shown in FIG0 .

Recent work has explored more advanced structures for continual learning BID40 and multi-task learning more generally BID48 BID37 .

These architectural advances are complementary to the new learning schemes developed here and a synthesis of the two would be potentially more powerful.

Moreover, a general solution to continual learning would perform automatic continual model building adding new bespoke structure to the existing model as new tasks are encountered.

Although this is a very interesting research direction, here we make the simplifying assumption that the model structure is known a priori.

VCL requires specification of q(θ) where θ in the current case is a D dimensional vector formed by stacking the network's biases and weights.

For simplicity we use a Gaussian mean-field approximate posterior DISPLAYFORM0 Taking the most general case of a multi-head network, before task k is encountered the posterior distribution over the associated head parameters will remain at the prior and so q(θ DISPLAYFORM1 .

This is convenient as it means the variational approximation can be grown incrementally, starting from the prior, as each task emerges.

Moreover, only tasks present in the current dataset D t need to have their posterior distributions over head parameters updated.

The shared parameters, on the other hand, will be constantly updated.

Training the network using the VFE approach in eq. (1) is equivalent to maximizing the negative online variational free energy or the variational lower bound to the online marginal likelihood DISPLAYFORM2 with respect to the variational parameters DISPLAYFORM3 .

Whilst the KL-divergence KL(q t (θ)||q t−1 (θ)) can be computed in closed-form, the expected log-likelihood requires further approximation.

Here we take the usual approach of employing simple Monte Carlo and use the local reparameterization trick to compute the gradients BID41 BID23 .

At the first time step, the prior distribution, and therefore q 0 (θ) is chosen to be a multivariate Gaussian distribution (see e.g. BID15 BID3 .

Deep generative models (DGMs) have garnered much recent attention.

By passing a simple noise variable (e.g. Gaussian noise) through a deep neural network, these models have been shown to be able to generate realistic images, sounds and videos sequences BID8 BID25 BID49 .

Standard approaches for learning DGMs have focused on batch learning, i.e. the observed instances are assumed to be i.i.d.

and are all available at the same time.

In this section we extend the VCL framework to encompass variational auto-encoders (VAEs) BID23 BID38 , a form of DGM.

The approach could be extended to generative adversarial networks (GANs) BID14 for which continual learning is an open problem (see BID44 for an initial attempt).Consider a model p(x|z, θ)p(z), for observed data x and latent variables z.

The prior over latent variables p(z) is typically Gaussian, and the distributional parameters of p(x|z, θ) are defined by a deep neural network.

For example, if Bernoulli likelihood is used, then p(x|z, θ) = Bern(x; f θ (z)), where f θ denotes the deep neural network transform and θ collects all the weight matrices and bias vectors.

In the batch setting, given a dataset DISPLAYFORM0 , the standard VAE approach learns the parameters θ by approximate maximum likelihood estimation (MLE).

This proceeds by maximizing the variational lower bound with respect to θ and φ: DISPLAYFORM1 where φ are the variational parameters of the approximate posterior or "encoder".The approximate MLE approach is unsuitable for the continual learning setting as it does not return parameter uncertainty estimates that are critical for weighting the information learned from old data.

So, instead the VCL approach will approximate the full posterior distribution over parameters, q t (θ) ≈ p(θ|D 1:t ), after observing the t-th dataset.

Specifically, the approximate posterior q t is obtained by maximizing the full variational lower bound with respect to q t and φ: DISPLAYFORM2 where the encoder network q φ (z DISPLAYFORM3 is parameterized by φ which is task-specific.

It is likely to be beneficial to share (parts of) these encoder networks, but this is not investigated in this paper.

As was the case for multi-head discriminative models, we can split the generative model into shared and task-specific parts.

There are two options: (i) the generative models share across tasks the network that generates observations x from the intermediate-level representations h, but have private "head networks" for generating h from the latent variables z (see FIG0 ), and (ii) the other way around.

Architecture (i) is arguably more appropriate when data are composed of a common set of structural primitives (such as strokes in handwritten digits) that are selected by high level variables (character identities).

Moreover, initial experiments on architecture (ii) indicated that information about the current task tended to be encoded entirely in the task-specific lower-level network negating multi-task transfer.

For these reasons, we focus on architecture (i) in the experiments.

Continual Learning for Deep Discriminative Models: Many neural network continual learning approaches employ regularized maximum likelihood estimation, optimizing objectives of the form: DISPLAYFORM0 .

Here the regularization biases the new parameter estimates towards those estimated at the previous step θ t−1 .

λ t is a user-selected hyper-parameter that controls the overall contribution from previous data and Σ t−1 is a matrix (normally diagonal in form) that encodes the relative strength of the regularization on each element of θ.

We now discuss specific instances of this scheme:• Maximum-likelihood estimation and MAP estimation: maximum likelihood estimation is recovered when there is no regularization (λ t = 0).

More generally, the regularization term can be interpreted as a Gaussian prior, q(θ|D 1:t−1 ) = N (θ; θ t−1 , Σ t−1 /λ t ).

The optimization returns the maximum a posteriori estimate of the parameters, but this does not directly provide Σ t for the next stage.

A simple fix is to set Σ t = I and use cross-validation to find λ t , but this approximation is often coarse and leads to catastrophic forgetting BID13 BID26 ).•

Laplace Propagation (LP) BID46 : applying Laplace's approximation at each step leads to a recursion for Σ −1 t , which is initialized using the covariance of the Gaussian prior, Σ −1 DISPLAYFORM1 To avoid computing the full Hessian of the likelihood, diagonal Laplace propagation retains only the diagonal terms of Σ −1 t .Published as a conference paper at ICLR 2018• Elastic Weight Consolidation (EWC) BID26 builds on diagonal Laplace propagation by approximating the average Hessian of the likelihoods using well-known identities for the Fisher information: DISPLAYFORM2 EWC also modifies the Laplace regularization, DISPLAYFORM3 , introducing hyper-parameters, removing the prior and regularizing to intermediate parameter estimates, rather than just those derived from the last task, DISPLAYFORM4 These changes may be unnecessary BID20 BID21 and require storing θ 1:t−1 , but may slightly improve performance (see BID27 and our experiments).• Synaptic Intelligence (SI) BID50 : SI computes Σ −1 t using a measure of the importance of each parameter to each task.

Practically, this is achieved by comparing the changing rate of the gradients of the objective and the changing rate of the parameters.

VCL differs from the above methods in several ways.

First, unlike MAP, EWC and SI, it does not have free parameters that need to be tuned on a validation set.

This can be especially awkward in the online setting.

Second, although the KL regularization penalizes the mean of the approximate posterior through a quadratic cost, a full distribution is retained and averaged over at training time and at test time.

Third, VI is generally thought to return better uncertainty estimates than approaches like Laplace's method and MAP estimation, and we have argued this is critical for continual learning.

There is a long history of research on approximate Bayesian training of neural networks, including extended Kalman filtering BID45 ), Laplace's approximation BID32 , variational inference BID18 BID2 BID15 BID3 BID10 , sequential Monte Carlo BID9 , expectation propagation (EP) BID16 , and approximate power EP .

These approaches have focused on batch learning, but the framework described in section 2 enables them to be applied to continual learning.

On the other hand, online variational inference has been previously explored BID11 BID4 BID6 , but not for neural networks or in the context of sets of related complex tasks.

Continual Learning for Deep Generative Models: A naïve continual learning approach for deep generative models would directly apply the VAE algorithm to the new dataset D t with the model parameters initialized at the previous parameter values θ t−1 .

The experiments show that this approach leads to catastrophic forgetting, in the sense that the generator can only generate instances that are similar to the data points from the most recently observed task.

Alternatively, EWC regularization can be added to the VAE objective: DISPLAYFORM5 However computing Φ t requires the gradient of the intractable marginal likelihood ∇ θ log p(x|θ).

Instead, we can approximate the marginal likelihood by the variational lower bound, i.e. DISPLAYFORM6 Similar variational lower-bound approximations apply when computing the Hessian matrices for LP and Σ −1 t for SI.

An importance sampling estimate could also be used BID7 .

The experiments evaluate the performance and flexibility of VCL through three discriminative tasks and two generative tasks.

Standard continual learning benchmarks are used where possible.

Comparisons are made to EWC, diagonal LP and SI that employ tuned hyper-parameters λ whereas VCL's objective is hyper-parameter free.

More details of the experiment settings and an additional experiment are available in the appendix.

We consider the following three continual learning experiments for deep discriminative models.

Permuted MNIST: This is a popular continual learning benchmark BID13 BID26 BID50 .

The dataset received at each time step D t consists of labeled MNIST images whose pixels have undergone a fixed random permutation.

We compare VCL to EWC, SI, and diagonal LP.

For all algorithms, we use fully connected single-head networks with two hidden layers, where each layer contains 100 hidden units with ReLU activations.

We evaluate three versions of VCL: VCL with no coreset, VCL with a random coreset, and VCL with a coreset selected by the K-center method.

For the coresets, we select 200 data points from each task.

FIG1 compares the average test set accuracy on all observed tasks.

From this figure, VCL outperforms EWC, SI, and LP by large margins, even though they benefited from an extensive hyperparameter search for λ.

Diagonal LP performs slightly worse than EWC both when λ = 1 and when the values of λ are tuned.

After 10 tasks, VCL achieves 90% average accuracy, while EWC, SI, and LP only achieve 84%, 86%, and 82% respectively.

The results also show that the coresets perform poorly by themselves, but combining them with VCL leads to a modest improvement: both random coresets and K-center coresets achieve 93% accuracy.

We also investigate the effect of the coreset size.

In FIG3 , we plot the average test set accuracy of VCL with random coresets of different sizes.

At the coreset size of 5,000 examples per task, VCL achieves 95.5% accuracy after 10 tasks, which is significantly better than the 90% of vanilla VCL.

Performance improves with the coreset size although it asymptotes for large coresets as expected: if a sufficiently large coreset is employed, it will be fully representative of the task and thus training on the coreset alone can achieve a good performance.

However, the experiments show that the combination of VCL and coresets is advantageous even for large coresets.

Split MNIST: This experiment was used by BID50 to assess the SI method.

Five binary classification tasks from the MNIST dataset arrive in sequence: 0/1, 2/3, 4/5, 6/7, and 8/9.

We use fully connected multi-head networks with two hidden layers comprising 256 hidden units with ReLU activations.

We compare VCL (with and without coresets) to EWC, SI, and diagonal LP.

For the coresets, 40 data points from each task are selected through random sampling or the K-center method.

FIG4 compares the test set accuracy on individual tasks (averaged over 10 runs) as well as the accumulated accuracy averaged over tasks (right).

As an upper bound on the algorithms' performance, we compare to batch VI trained on the full dataset.

From this figure, VCL significantly outperforms EWC and LP although it is slightly worse than SI.

Again, unlike VCL, EWC and SI benefited from a hyper-parameter search for λ, but a value close to 1 performs well in both cases.

After 5 tasks, VCL achieves 97.0% average accuracy on all tasks, while EWC, SI, and LP attain 63.1%, 98.9%, and 61.2% respectively.

Adding the coreset improves VCL to around 98.4% accuracy.

This experiment is similar to the previous one, but it uses the more challenging notMNIST dataset and deeper networks.

The notMNIST dataset 2 here contains 400,000 images of the characters from A to J with different font styles.

We consider five binary classification tasks: A/F, B/G, C/H, D/I, and E/J using deeper networks comprising four hidden layers of 150 hidden units with ReLU activations.

The other settings are kept the same as the previous experiment.

VCL is competitive with SI and significantly outperforms EWC and LP (see FIG5 ), although the SI and EWC baselines benefited from a hyper-parameter search.

VCL achieves 92.0% average accuracy after 5 tasks, while EWC, SI, and LP attain 71%, 94%, and 63% respectively.

Adding the random coreset improves the performance of VCL to 96% accuracy.

We consider two continual learning experiments for deep generative models: MNIST digit generation and notMNIST (small) character generation.

In both cases, ten datasets are received in sequence.

For MNIST, the first dataset comprises exclusively of images of the digit zero, the second dataset ones and so on.

For notMNIST, the datasets contain the characters A to J in sequence.

The generative model consists of shared and task-specific components, each represented by a one hidden layer neural network with 500 hidden units (see FIG0 ).

The dimensionality of the latent variable z and the intermediate representation h are 50 and 500, respectively.

We use task-specific encoders that are neural networks with symmetric architectures to the generator.

We compare VCL to naïve online learning using the standard VAE objective, LP, EWC and SI (with hyper-parameters λ = 1, 10, 100).

For full details of the experimental settings see Appendix E. Samples from the generative models attained at different time steps are shown in fig. 6 .

The naïve online learning method fails catastrophically and so numerical results are omitted.

LP, EWC, SI and VCL remember previous tasks, with SI and VCL achieving the best visual quality on both datasets.

The algorithms are quantitatively evaluated using two metrics in FIG7 : an importance sampling estimate of the test log-likelihood (test-LL) using 5, 000 samples and a measure of quality we term "classifier uncertainty".

For the latter, we train a discriminative classifier for the digits/alphabets to achieve high accuracy.

The quality of generated samples can then be assessed by the KL-divergence from the one-hot vector indicating the task, to the output classification probability vector computed on the generated images.

A well-trained generator will produce images that are correctly classified in high confidence resulting in zero KL.

We only report the best performance for LP, EWC and SI.We observe that LP and EWC perform similarly, most likely due to the fact that both LP and EWC use the same Σ t matrices.

EWC achieves significantly worse performance than SI.

VCL is on par with or slightly better than SI.

VCL has a superior long-term memory of previous tasks which leads to better overall performance on both metrics even though it does not have tuned hyper-parameters in its objective function.

For MNIST, the performance of LP and EWC deteriorate markedly when moving from task "digit 0" to "digit 1" possibly due to the large task differences.

Also for all experimental settings we tried, SI fails to produce high test-LL results after task "digit 7".

Future work will investigate continual learning on a sequence of tasks that follows "adversarial ordering", i.e. the ordering that makes the next task maximally different from the current task.

Approximate Bayesian inference provides a natural framework for continual learning.

Variational Continual Learning (VCL), developed in this paper, is an approach in this vein that extends online variational inference to handle more general continual learning tasks and complex neural network models.

VCL can be enhanced by including a small episodic memory that leverages coreset algorithms from statistics and connects to message-scheduling in variational message passing.

We demonstrated how the VCL framework can be applied to both discriminative and generative models.

Experimental results showed state-of-the-art performance when compared to previous continual learning approaches, even though VCL has no free parameters in its objective function.

Future work should explore alternative approximate inference methods using the same framework and also develop more sophisticated episodic memories.

Finally, we note that VCL is ideally suited for efficient model refinement in sequential decision making problems, such as reinforcement learning and active learning.

DISPLAYFORM0 Figure 6: Generated images from each of the generators after training.

Each of the columns shows the images generated from a specific task's generator, and each of the lines shows the generations from generators of all trained tasks.

Clearly the naive approach suffers from catastrophic forgetting, while other approaches successfully remember previous tasks.

In this experiment, we use fully connected single-head networks with two hidden layers, where each layer contains 100 hidden units with ReLU activations.

The metric used for comparison is the test set accuracy on all observed tasks.

We train all the models using the Adam optimizer BID22 with learning rate 10 −3 since we found that it works best for all models.

All the VCL algorithms are trained with batch size 256 and 100 epochs.

For all the algorithms with coresets, we choose 200 examples from each task to include into the coresets.

The algorithms that use only the coresets are trained using the VFE method with batch size equal to the coreset size and 100 epochs.

We use the prior N (0, I) and initialize our optimizer for the first task at the mean of the maximum likelihood model and a very small initial variance (10 −6 ).We compare the performance of SI with hyper-parameters λ = 0.01, 0.1, 0.5, 1, 2 and select the best one (λ = 0.5) as our baseline (see fig. 8 ).

Following BID50 , we train these models with batch size 256 and 20 epochs.

We also compare the performance of EWC with λ = 1, 10, 10 2 , 10 3 , 10 4 and select the best value λ = 10 2 as our baseline (see fig. 9 ).

The models are trained without dropout and with batch size 200 and 20 epochs.

We approximate the Fisher information matrices in EWC using 600 random samples drawn from the current dataset.

For diagonal LP, we compare the performance of λ = 0.01, 0.1, 1, 10, 100 and use the best value λ = 0.1 as our baseline (see FIG0 ).

The models are also trained with prior N (0, I), batch size 200, and 20 epochs.

The Hessians of LP are approximated using the Fisher information matrices with 200 samples.

In this experiment, we use fully connected multi-head networks with two hidden layers, each of which contains 256 hidden units with ReLU activations.

At each time step, we compare the test set accuracy of the current model on all observed tasks separately.

We also plot the average accuracy over all tasks in the last column of FIG4 .

All the results for this experiment are the averages over 10 runs of the algorithms with different random seeds.

We use the Adam optimizer with learning rate 10 −3 for all models.

All the VCL algorithms are trained with batch size equal to the size of the training set and 120 epochs.

We use the prior N (0, I) and initialize our optimizer for the first task at the mean of the maximum likelihood model and a very small initial variance (10 −6 ).

For the coresets, we choose 40 examples from each task to include into the coresets.

In this experiment, the final approximate posterior used for prediction in eq. (3) is computed for each task separately using the coreset points corresponding to the task.

The algorithms that use only the coresets are trained using the VFE method with batch size equal to the coreset size and 120 epochs.

We compare the performance of SI with λ = 0.01, 0.1, 1, 2, 3 and use the best value λ = 1 as our baseline (see FIG0 ).

We also compare EWC with both single-head and multi-head models and λ = 1, 10, 10 2 , 10 3 , 10 4 (see FIG0 ).

We approximate the Fisher information matrices using 200 random samples drawn from the current dataset.

The figure shows that the multi-head models work better than the single-head models for EWC, and the performance is insensitive to the choice of λ.

Thus, we use the multi-head model with λ = 1 as the EWC baseline for our experiment.

For diagonal LP, we also use the multi-head model with λ = 1, prior N (0, I), and approximate the Hessians using the Fisher information matrices with 200 samples.

The settings for this experiment are the same as those in the Split MNIST experiment above, except that we use deeper networks with 4 hidden layers, each of which contains 150 hidden units.

FIG0 show the performance of SI and EWC with different hyper-parameter values respectively.

In the experiment, we choose λ = 10 4 for multi-head EWC, λ = 1 for multi-head LP, and λ = 0.1 for SI.

Here we consider a small experiment on a toy 2D dataset to understand some of the properties of EWC with λ = 1 and VCL.

The experiment comprises two sequential binary classification tasks.

The first task contains two classes generated from two Gaussian distributions.

The data points (with green and black classes) for this task are shown in the first column of FIG0 .

The second task contains two classes also generated from two Gaussian distributions.

The green class for this task has the same input distribution as the first task, while the input distribution for the black class is different.

The data points for this task are shown in the third columns of FIG0 .

Each task contains 200 data points with 100 data points in each class.

We compare the multi-head models trained by VCL and EWC on these two tasks.

In this experiment, we use fully connected networks with one hidden layer containing 20 hidden units with ReLU activations.

The first column of FIG0 shows the contours of the prediction probabilities after observing the first task, and both methods perform reasonably well for this task.

However, after observing the second task, the EWC method fails to learn the classifiers for both tasks, while the VCL method are still able to learn good classifiers for them.

In the experiments on Deep Generative Models, the learning rates and numbers of optimization epochs are tuned on separate training of each tasks.

This gives a learning rate of 10 −4 and the number of epochs 200 for MNIST (except for SI) and 400 for notMNIST.

For SI we optimize for 400 epochs on MNIST.

For the VCL approach, the parameters of q t (θ) are initialized to have the same mean as q t−1 (θ) and the log standard deviation is set to 10 −6 .The generative model consists of shared and task-specific components, each represented by a one hidden layer neural network with 500 hidden units (see FIG0 ).

The dimensionality of the latent variable z and the intermediate representation h are 50 and 500, respectively.

We use task-specific encoders that are neural networks with symmetric architectures to the generator.

In many probabilistic models with conjugate priors, the exact posterior of the parameters/latent variables can be obtained.

For example, a Bayesian linear regression model with a Gaussian prior over the parameters and a Gaussian observation model has a Gaussian posterior.

If we insist on using a diagonal Gaussian approximation to this posterior and use either the variational free energy method or the Laplace's approximation, we will end up at the same solution -a Gaussian distribution with the same mean as that of the exact posterior and the diagonal precisions being the diagonal precisions of the exact posterior.

Consequently, the online variational Gaussian approximation will give the same result to that given by the online Laplace's approximation.

However, when a diagonal Gaussian approximation is used, the batch and sequential solutions are different.

In the following, we will explicitly detail the sequential variational updates for a Bayesian linear regression model to associate random binary patterns to binary outcomes BID26 , and show its relationship to the online Laplace's approximation and the EWC approach of BID26 .

The task consists of associating a random D-dimensional binary vector x t to a random binary output y t by learning a weight vector W .

Note that the possible values of the features and outputs are 1 and −1, and not 0 and 1.

We also assume that the model sees only one input-output pair {x t , y t } at the t-th time step and the previous approximate posterior q t−1 (W ) = By further assuming that σ y = 1 and x t 2 = 1, the equations above become: DISPLAYFORM0 DISPLAYFORM1 wherex t = y t x t .

When v −1 0,d = 0, i.e. the prior is ignored, the update for the mean above is exactly equation S4 in the supplementary material of BID26 .

Therefore, in this case, the memory of the network trained by online variational inference is identical to that of the online Laplace's method, provided in BID26 .

These methods differ, in practice, when the prior is not ignored or when the parameter regularization constraints are accumulated as discussed in the main text.

This equivalence also does not hold in the general case, as discussed by BID35 where the Gaussian variational approximation can be interpreted as an averaged and smoothed Laplace's approximation.

<|TLDR|>

@highlight

This paper develops a principled method for continual learning in deep models.