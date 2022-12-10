Energy  based  models  outputs  unmormalized  log-probability  values  given  datasamples.

Such a estimation is essential in a variety of application problems suchas sample generation, denoising, sample restoration, outlier detection, Bayesianreasoning, and many more.

However, standard maximum likelihood training iscomputationally expensive due to the requirement of sampling model distribution.

Score matching potentially alleviates this problem, and denoising score matching(Vincent, 2011) is a particular convenient version.

However,  previous attemptsfailed to produce models capable of high quality sample synthesis.

We believethat  it  is  because  they  only  performed  denoising  score  matching  over  a  singlenoise scale.

To overcome this limitation, here we instead learn an energy functionusing all noise scales.

When sampled using Annealed Langevin dynamics andsingle step denoising jump, our model produced high-quality samples comparableto state-of-the-art techniques such as GANs, in addition to assigning likelihood totest data comparable to previous likelihood models.

Our model set a new sam-ple quality baseline in likelihood-based models.

We further demonstrate that our model learns sample distribution and generalize well on an image inpainting tasks.

Treating data as stochastic samples from a probability distribution and developing models that can learn such distributions is at the core for solving a large variety of application problems, such as error correction/denoising (Vincent et al., 2010) , outlier/novelty detection (Zhai et al., 2016; Choi and Jang, 2018) , sample generation (Nijkamp et al., 2019; Du and Mordatch, 2019) , invariant pattern recognition, Bayesian reasoning (Welling and Teh, 2011) which relies on good data priors, and many others.

Energy-Based Models (EBMs) (LeCun et al., 2006; Ngiam et al., 2011 ) assign an energy E(x x x) to each data point x x x which implicitly defines a probability by the Boltzmann distribution p m (x x x) = e −E(x x x) /Z. Sampling from this distribution can be used as a generative process that yield plausible samples of x x x. Compared to other generative models, like GANs (Goodfellow et al., 2014) , flowbased models (Dinh et al., 2015; Kingma and Dhariwal, 2018) , or auto-regressive models (van den Oord et al., 2016; Ostrovski et al., 2018) , energy-based models have significant advantages.

First, they provide explicit (unnormalized) density information, compositionality (Hinton, 1999; Haarnoja et al., 2017) , better mode coverage (Kumar et al., 2019) and flexibility (Du and Mordatch, 2019) .

Further, they do not require special model architecture, unlike auto-regressive and flow-based models.

Recently, Energy-based models has been successfully trained with maximum likelihood (Nijkamp et al., 2019; Du and Mordatch, 2019) , but training can be very computationally demanding due to the need of sampling model distribution.

Variants with a truncated sampling procedure have been proposed, such as contrastive divergence (Hinton, 2002) .

Such models learn much faster with the draw back of not exploring the state space thoroughly (Tieleman, 2008) .

Score matching (SM) (Hyvärinen, 2005) circumvents the requirement of sampling the model distribution.

In score matching, the score function is defined to be the gradient of log-density or the negative energy function.

The expected L2 norm of difference between the model score function and the data score function are minimized.

One convenient way of using score matching is learning the energy function corresponding to a Gaussian kernel Parzen density estimator (Parzen, 1962) of the data: p σ0 (x x x) = q σ0 (x x x|x x x)p(x x x)dx x x. Though hard to evaluate, the data score is well defined: s d (x x x) = ∇x x x log(p σ0 (x x x)), and the corresponding objective is:

L SM (θ) = E pσ0(x x x) ∇x x x log(p σ0 (x x x)) + ∇x x x E(x x x; θ)

(1) Vincent (2011) studied the connection between denoising auto-encoder and score matching, and proved the remarkable result that the following objective, named Denoising Score Matching (DSM), is equivalent to the objective above:

L DSM (θ) = E pσ 0 (x x x,x x x) ∇x x x log(q σ0 (x x x|x x x)) + ∇x x x E(x x x; θ)

Note that in (2) the Parzen density score is replaced by the derivative of log density of the single noise kernel ∇x x x log(q σ0 (x x x|x x x)), which is much easier to evaluate.

In the particular case of Gaussian noise, log(q σ0 (x x x|x x x)) = − (x x x−x x x) 2 2σ 2 0 + C, and therefore:

The interpretation of objective (3) is simple, it forces the energy gradient to align with the vector pointing from the noisy sample to the clean data sample.

To optimize an objective involving the derivative of a function defined by a neural network, Kingma and LeCun (2010) proposed the use of double backpropagation (Drucker and Le Cun, 1991) .

Deep energy estimator networks (Saremi et al., 2018) first applied this technique to learn an energy function defined by a deep neural network.

In this work and similarly in Saremi and Hyvarinen (2019) , an energy-based model was trained to match a Parzen density estimator of data with a certain noise magnitude.

The previous models were able to perform denoising task, but they were unable to generate high-quality data samples from a random input initialization.

Recently, trained an excellent generative model by fitting a series of score estimators coupled together in a single neural network, each matching the score of a Parzen estimator with a different noise magnitude.

The questions we address here is why learning energy-based models with single noise level does not permit high-quality sample generation and what can be done to improve energy based models.

Our work builds on key ideas from Saremi et al. (2018) ; Saremi and Hyvarinen (2019) ; .

Section 2, provides a geometric view of the learning problem in denoising score matching and provides a theoretical explanation why training with one noise level is insufficient if the data dimension is high.

Section 3 presents a novel method for training energy based model, Multiscale Denoising Score Matching (MDSM).

Section 4 describes empirical results of the MDSM model and comparisons with other models.

2 A GEOMETRIC VIEW OF DENOISING SCORE MATCHING used denoising score matching with a range of noise levels, achieving great empirical results.

The authors explained that large noise perturbation are required to enable the learning of the score in low-data density regions.

But it is still unclear why a series of different noise levels are necessary, rather than one single large noise level.

Following Saremi and Hyvarinen (2019) , we analyze the learning process in denoising score matching based on measure concentration properties of high-dimensional random vectors.

We adopt the common assumption that the data distribution to be learned is high-dimensional, but only has support around a relatively low-dimensional manifold (Tenenbaum et al., 2000; Roweis and Saul, 2000; Lawrence, 2005) .

If the assumption holds, it causes a problem for score matching:

The density, or the gradient of the density is then undefined outside the manifold, making it difficult to train a valid density model for the data distribution defined on the entire space.

Saremi and Hyvarinen (2019) and discussed this problem and proposed to smooth the data distribution with a Gaussian kernel to alleviate the issue.

To further understand the learning in denoising score matching when the data lie on a manifold X and the data dimension is high, two elementary properties of random Gaussian vectors in highdimensional spaces are helpful: First, the length distribution of random vectors becomes concentrated at √ dσ (Vershynin, 2018) , where σ 2 is the variance of a single dimension.

Second, a random vector is always close to orthogonal to a fixed vector (Tao, 2012) .

With these premises one can visualize the configuration of noisy and noiseless data points that enter the learning process: A data point x x x sampled from X and its noisy versionx x x always lie on a line which is almost perpendicular to the tangent space T x x x X and intersects X at x x x. Further, the distance vectors between (x x x,x x x) pairs all have similar length √ dσ.

As a consequence, the set of noisy data points concentrate on a setX √

that has a distance with (

Therefore, performing denoising score matching learning with (x x x,x x x) pairs generated with a fixed noise level σ, which is the approach taken previously except in , will match the score in the setX √ dσ, and enable denoising of noisy points in the same set.

However, the learning provides little information about the density outside this set, farther or closer to the data manifold, as noisy samples outsideX √ dσ, rarely appear in the training process.

An illustration is presented in Figure 1A .

) is very small in high-dimensional space, the score inX

still plays a critical role in sampling from random initialization.

This analysis may explain why models based on denoising score matching, trained with a single noise level encounter difficulties in generating data samples when initialized at random.

For an empirical support of this explanation, see our experiments with models trained with single noise magnitudes (Appendix B).

To remedy this problem, one has to apply a learning procedure of the sort proposed in , in which samples with different noise levels are used.

Depending on the dimension of the data, the different noise levels have to be spaced narrowly enough to avoid empty regions in the data space.

In the following, we will use Gaussian noise and employ a Gaussian scale mixture to produce the noisy data samples for the training (for details, See Section 3.1 and Appendix A).

Another interesting property of denoising score matching was suggested in the denoising autoencoder literature (Vincent et al., 2010; Karklin and Simoncelli, 2011) .

With increasing noise level, the learned features tend to have larger spatial scale.

In our experiment we observe similar phenomenon when training model with denoising score matching with single noise scale.

If one compare samples in Figure B .1, Appendix B, it is evident that noise level of 0.3 produced a model that learned short range correlation that spans only a few pixels, noise level of 0.6 learns longer stroke structure without coherent overall structure, and noise level of 1 learns more coherent long range structure without details such as stroke width variations.

This suggests that training with single noise level in denoising score matching is not sufficient for learning a model capable of high-quality sample synthesis, as such a model have to capture data structure of all scales.

Motivated by the analysis in section 2, we strive to develop an EBM based on denoising score matching that can be trained with noisy samples in which the noise level is not fixed but drawn from a distribution.

The model should approximate the Parzen density estimator of the data p σ0 (x x x) = q σ0 (x x x|x x x)p(x x x)dx.

Specifically, the learning should minimize the difference between the derivative of the energy and the score of p σ0 under the expectation E p M (x x x) rather than E pσ 0 (x x x) , the expectation taken in standard denoising score matching.

Here p M (x x x) = q M (x x x|x x x)p(x x x)dx is chosen to cover the signal space more evenly to avoid the measure concentration issue described above.

The resulting Multiscale Score Matching (MSM) objective is:

Compared to the objective of denoising score matching (1), the only change in the new objective (4) is the expectation.

Both objectives are consistent, if p M (x x x) and p σ0 (x x x) have the same support, as shown formally in Proposition 1 of Appendix A. In Proposition 2, we prove that Equation 4 is equivalent to the following denoising score matching objective:

The above results hold for any noise kernel q σ0 (x x x|x x x), but Equation 5 contains the reversed expectation, which is difficult to evaluate in general.

To proceed, we choose q σ0 (x x x|x x x) to be Gaussian, and also choose q M (x x x|x x x) to be a Gaussian scale mixture: q M (x x x|x x x) = q σ (x x x|x x x)p(σ)dσ and q σ (x x x|x x x) = N (x x x, σ 2 I d ).

After algebraic manipulation and one approximation (see the derivation following Proposition 2 in Appendix A), we can transform Equation 5 into a more convenient form, which we call Multiscale Denoising Score Matching (MDSM):

The square loss term evaluated at noisy pointsx x x at larger distances from the true data points x x x will have larger magnitude.

Therefore, in practice it is convenient to add a monotonically decreasing term l(σ) for balancing the different noise scales, e.g. l(σ) = 1 σ 2 .

Ideally, we want our model to learn the correct gradient everywhere, so we would need to add noise of all levels.

However, learning denoising score matching at very large or very small noise levels is useless.

At very large noise levels the information of the original sample is completely lost.

Conversely, in the limit of small noise, the noisy sample is virtually indistinguishable from real data.

In neither case one can learn a gradient which is informative about the data structure.

Thus, the noise range needs only to be broad enough to encourage learning of data features over all scales.

Particularly, we do not sample σ but instead choose a series of fixed σ values

, we arrive at the final objective:

It may seem that σ 0 is an important hyperparameter to our model, but after our approximation σ 0 become just a scaling factor in front of the energy function, and can be simply set to one as long as the temperature range during sampling is scaled accordingly (See Section 3.2).

Therefore the only hyper-parameter is the rang of noise levels used during training.

On the surface, objective (7) looks similar to the one in .

The important difference is that Equation 7 approximates a single distribution, namely p σ0 (x x x), the data smoothed with one fixed kernel q σ0 (x x x|x x x).

In contrast, approximate the score of multiple distributions, the family of distributions {p σi (x x x) : i = 1, ..., n}, resulting from the data smoothed by kernels of different widths σ i .

Because our model learns only a single target distribution, it does not require noise magnitude as input.

Langevin dynamics has been used to sample from neural network energy functions (Du and Mordatch, 2019; Nijkamp et al., 2019) .

However, the studies described difficulties with mode exploration unless very large number of sampling steps is used.

To improve mode exploration, we propose incorporating simulated annealing in the Langevin dynamics.

Simulated annealing (Kirkpatrick et al., 1983; Neal, 2001) improves mode exploration by sampling first at high temperature and then cooling down gradually.

This has been successfully applied to challenging computational problems, such as combinatorial optimization.

To apply simulated annealing to Langevin dynamics.

Note that in a model of Brownian motion of a physical particle, the temperature in the Langevin equation enters as a factor √ T in front of the noise term, some literature uses β −1 where β = 1/T (Jordan et al., 1998) .

Adopting the √ T convention, the Langevin sampling process (Bellec et al., 2017 ) is given by:

where T t follows some annealing schedule, and denotes step length, which is fixed.

During sampling, samples behave very much like physical particles under Brownian motion in a potential field.

Because the particles have average energies close to the their current thermic energy, they explore the state space at different distances from data manifold depending on temperature.

Eventually, they settle somewhere on the data manifold.

The behavior of the particle's energy value during a typical annealing process is depicted in Appendix Figure F .1B.

If the obtained sample is still slightly noisy, we can apply a single step gradient denoising jump (Saremi and Hyvarinen, 2019) to improve sample quality:

This denoising procedure can be applied to noisy sample with any level of Gaussian noise because in our model the gradient automatically has the right magnitude to denoise the sample.

This process is justified by the Empirical Bayes interpretation of this denoising process, as studied in Saremi and Hyvarinen (2019) .

Song and Ermon (2019) also call their sample generation process annealed Langevin dynamics.

It should be noted that their sampling process does not coincide with Equation 8.

Their sampling procedure is best understood as sequentially sampling a series of distributions corresponding to data distribution corrupted by different levels of noise.

Training and Sampling Details.

The proposed energy-based model is trained on standard image datasets, specifically MNIST, Fashion MNIST, CelebA (Liu et al., 2015) and CIFAR-10 (Krizhevsky et al., 2009) .

During training we set σ 0 = 0.1 and train over a noise range of σ ∈ [0.05, 1.2], with the different noise uniformly spaced on the batch dimension.

For MNIST and Fashion MNIST we used geometrically distributed noise in the range [0.1, 3].

The weighting factor l(σ) is always set to 1/σ 2 to make the square term roughly independent of σ.

We fix the batch size at 128 and use the Adam optimizer with a learning rate of 5 × 10 −5 .

For MNIST and Fashion MNIST, we use a 12-Layer ResNet with 64 filters, for the CelebA and CIFAT-10 data sets we used a 18-Layer ResNet with 128 filters (He et al., 2016a; .

No normalization layer was used in any of the networks.

We designed the output layer of all networks to take a generalized quadratic form (Fan et al., 2018) .

Because the energy function is anticipated to be approximately quadratic with respect to the noise level, this modification was able to boost the performance significantly.

For more detail on training and model architecture, see Appendix D. One notable result is that since our training method does not involve sampling, we achieved a speed up of roughly an order of magnitude compared to the maximum-likelihood training using Langevin dynamics 1 .

Our method thus enables the training of energy-based models even when limited computational resources prohibit maximum likelihood methods.

We found that the choice of the maximum noise level has little effect on learning as long as it is large enough to encourage learning of the longest range features in the data.

However, as expected, learning with too small or too large noise levels is not beneficial and can even destabilize the training process.

Further, our method appeared to be relatively insensitive to how the noise levels are distributed over a chosen range.

Geometrically spaced noise as in and linearly spaced noise both work, although in our case learning with linearly spaced noise was somewhat more robust.

For sampling the learned energy function we used annealed Langevin dynamics with an empirically optimized annealing schedule,see Figure F .1B for the particular shape of annealing schedule we used.

In contrast, annealing schedules with theoretical guaranteed convergence property takes extremely long (Geman and Geman, 1984) .

The range of temperatures to use in the sampling process depends on the choice of σ 0 , as the equilibrium distribution is roughly images with Gaussian noise of magnitude √ T σ 0 added on top.

To ease traveling between modes far apart and ensure even sampling, the initial temperature needs to be high enough to inject noise of sufficient magnitude.

A choice of T = 100, which corresponds to added noise of magnitude √ 100 * 0.1 = 1, seems to be sufficient starting point.

For step length we generally used 0.02, although any value within the range [0.015, 0.05] seemed to work fine.

After the annealing process we performed a single step denoising to further enhance sample quality. (Salimans et al., 2016) and FID (Heusel et al., 2017) .

We achieved Inception Score of 8.31 and FID of 31.7, comparable to modern GAN approaches.

Scores for CelebA dataset are not reported here as they are not commonly reported and may depend on the specific pre-processing used.

More samples and training images are provided in Appendix for visual inspection.

We believe that visual assessment is still essential because of the possible issues with the Inception score (Barratt and Sharma, 2018) .

Indeed, we also found that the visually impressive samples were not necessarily the one achieving the highest Inception Score.

Although overfitting is not a common concern for generative models, we still tested our model for overfitting.

We found no indication for overfitting by comparing model samples with their nearest neighbors in the data set, see Figure C .1 in Appendix.

Mode Coverage.

We repeated with our model the 3 channel MNIST mode coverage experiment similar to the one in Kumar et al. (2019) .

An energy-based model was trained on 3-channel data where each channel is a random MNIST digit.

Then 8000 samples were taken from the model and each channel was classified using a small MNIST classifier network.

We obtained results of the 966 modes, comparable to GAN approaches.

Training was successful and our model assigned low energy to all the learned modes, but some modes were not accessed during sampling, likely due to the Langevin Dynamics failing to explore these modes.

A better sampling technique such as HMC Neal et al. (2011) or a Maximum Entropy Generator (Kumar et al., 2019) could improve this result.

Image Inpainting.

Image impainting can be achieved with our model by clamping a part of the image to ground truth and performing the same annealed Langevin and Jump sampling procedure on the missing part of the image.

Noise appropriate to the sampling temperature need to be added to the clamped inputs.

The quality of inpainting results of our model trained on CelebA and CIFAR-10 can be assessed in Figure 3 .

For CIFAR-10 inpainting results we used the test set.

Log likelihood estimation.

For energy-based models, the log density can be obtained after estimating the partition function with Annealed Importance Sampling (AIS) (Salakhutdinov and Murray, 2008) or Reverse AIS (Burda et al., 2015) .

In our experiment on CIFAR-10 model, similar to reports in Du and Mordatch (2019) , there is still a substantial gap between AIS and Reverse AIS estimation, even after very substantial computational effort.

In Table 1 We also report a density of 1.21 bits/dim on MNIST dataset, and we refer readers to Du and Mordatch (2019) for comparison to other models on this dataset.

More details on this experiment is provided in the Appendix.

Outlier Detection.

Choi and Jang (2018) and Nalisnick et al. (2019) have reported intriguing behavior of high dimensional density models on out of distribution samples.

Specifically, they showed that a lot of models assign higher likelihood to out of distribution samples than real data samples.

We investigated whether our model behaves similarly.

Our energy function is only trained outside the data manifold where samples are noisy, so the energy value at clean data points may not always be well behaved.

Therefore, we added noise with magnitude σ 0 before measuring the energy value.

We find that our network behaves similarly to previous likelihood models, it assigns lower energy, thus higher density, to some OOD samples.

We show one example of this phenomenon in Appendix Figure F .1A.

We also attempted to use the denoising performance, or the objective function to perform outlier detection.

Intriguingly, the results are similar as using the energy value.

Denoising performance seems to correlate more with the variance of the original image than the content of the image.

In this work we provided analyses and empirical results for understanding the limitations of learning the structure of high-dimensional data with denoising score matching.

We found that the objective function confines learning to a small set due to the measure concentration phenomenon in random vectors.

Therefore, sampling the learned distribution outside the set where the gradient is learned does not produce good result.

One remedy to learn meaningful gradients in the entire space is to use samples during learning that are corrupted by different amounts of noise.

Indeed, Song and Ermon (2019) applied this strategy very successfully.

The central contribution of our paper is to investigate how to use a similar learning strategy in EBMs.

Specifically, we proposed a novel EBM model, the Multiscale Denoising Score Matching (MDSM) model.

The new model is capable of denoising, producing high-quality samples from random noise, and performing image inpainting.

While also providing density information, our model learns an order of magnitude faster than models based on maximum likelihood.

Our approach is conceptually similar to the idea of combining denoising autoencoder and annealing (Geras and Sutton, 2015; Chandra and Sharma, 2014; Zhang and Zhang, 2018) though this idea was proposed in the context of pre-training neural networks for classification applications.

Previous efforts of learning energy-based models with score matching (Kingma and LeCun, 2010; were either computationally intensive or unable to produce high-quality samples comparable to those obtained by other generative models such as GANs.

Saremi et al. (2018) and Saremi and Hyvarinen (2019) trained energy-based model with the denoising score matching objective but the resulting models cannot perform sample synthesis from random noise initialization.

Recently, proposed the NCSN model, capable of high-quality sample synthesis.

This model approximates the score of a family of distributions obtained by smoothing the data by kernels of different widths.

The sampling in the NCSN model starts with sampling the distribution obtained with the coarsest kernel and successively switches to distributions obtained with finer kernels.

Unlike NCSN, our method learns an energy-based model corresponding to p σ0 (x x x) for a fixed σ 0 .

This method improves score matching in high-dimensional space by matching the gradient of an energy function to the score of p σ0 (x x x) in a set that avoids measure concentration issue.

All told, we offer a novel EBM model that achieves high-quality sample synthesis, which among other EBM approaches provides a new state-of-the art.

Compared to the NCSN model, our model is more parsimonious than NCSN and can support single step denoising without prior knowledge of the noise magnitude.

But our model performs sightly worse than the NCSN model, which could have several reasons.

First, the derivation of Equation 6 requires an approximation to keep the training procedure tractable, which could reduce the performance.

Second, the NCSNs output is a vector that, at least during optimization, does not always have to be the derivative of a scalar function.

In contrast, in our model the network output is a scalar function.

Thus it is possible that the NCSN model performs better because it explores a larger set of functions during optimization.

In this section, we provide a formal discussion of the MDSM objective and suggest it as an improved score matching formulation in high-dimensional space.

Vincent (2011) illustrated the connection between the model score −∇x x x E(x x x; θ) with the score of Parzen window density estimator ∇x x x log(p σ0 (x x x)).

Specifically, the objective is Equation 1 which we restate here:

Our key observation is: in high-dimensional space, due the concentration of measure, the expectation w.r.t.

p σ0 (x x x) over weighs a thin shell at roughly distance √ dσ to the empirical distribution p(x).

Though in theory this is not a problem, in practice this leads to results that the score are only well matched on this shell.

Based on this observation, we suggest to replace the expectation w.r.t.

p σ0 (x x x) with a distribution p σ (x x x) that has the same support as p σ0 (x x x) but can avoid the measure concentration problem.

We call this multiscale score matching and the objective is the following:

Given that p M (x x x) and p σ0 (x x x) has the same support, it's clear that L M SM = 0 would be equivalent to L SM = 0.

Due to the proof of the Theorem 2 in Hyvärinen (2005), we have

We follow the same procedure as in Vincent (2011) to prove this result.

Thus we have:

The above analysis applies to any noise distribution, not limited to Gaussian.

but L M DSM * has a reversed expectation form that is not easy to work with.

To proceed further we study the case where q σ0 (x x x|x x x) is Gaussian and choose q M (x x x|x x x) as a Gaussian scale mixture (Wainwright and Simoncelli, 2000) and p M (x x x) = q M (x x x|x x x)p(x x x)dx.

By Proposition 1 and Proposition 2, we have the following form to optimize:

To minimize Equation (*), we can use the following importance sampling procedure (Russell and Norvig, 2016): we can sample from the empirical distribution p(x x x), then sample the Gaussian scale mixture q M (x x x|x x x) and finally weight the sample by qσ 0 (x x x|x x x) q M (x x x|x x x) .

We expect the ratio to be close to 1 for the following reasons: Using Bayes rule, q σ0 (x x x|x x x) = p(x x x)qσ 0 (x x x|x x x) pσ 0 (x x x)

we can see that q σ0 (x x x|x x x) only has support on discret data points x x x, same thing holds for q M (x x x|x x x).

because inx x x is generated by adding Gaussian noise to real data sample, both estimators should give results highly concentrated on the original sample point x x x. Therefore, in practice, we ignore the weighting factor and use Equation 6.

Improving upon this approximation is left for future work.

To compare with previous method, we trained energy-based model with denoising score matching using one noise level on MNIST, initialized the sampling with Gaussian noise of the same level, and sampled with Langevin dynamics at T = 1 for 1000 steps and perform one denoise jump to recover the model's best estimate of the clean sample, see Figure B .1.

We used the same 12-layer ResNet as other MNIST experiments.

Models were trained for 100000 steps before sampling.

We demonstrate that the model does not simply memorize training examples by comparing model samples with their nearest neighbors in the training set.

We use Fashion MNIST for this demonstration because overfitting can occur there easier than on more complicated datasets, see Figure C we used a 18-layer ResNet with 128 filters on the first layer.

All network used the ELU activation function.

We did not use any normalization in the ResBlocks and the filer number is doubled at each downsampling block.

Details about the structure of our networks used can be found in our code release.

All mentioned models can be trained on 2 GPUs within 2 days.

Since the gradient of our energy model scales linearly with the noise, we expected our energy function to scale quadratically with noise magnitude.

Therefore, we modified the standard energy-based network output layer to take a flexible quadratic form (Fan et al., 2018 ):

where a i , c i , d i and b 1 , b 2 , b 3 are learnable parameters, and h i is the (flattened) output of last residual block.

We found this modification to significantly improve performance compared to using a simple linear last layer.

For CIFAR and CelebA results we trained for 300k weight updates, saving a checkpoint every 5000 updates.

We then took 1000 samples from each saved networks and used the network with the lowest FID score.

For MNIST and fashion MNIST we simply trained for 100k updates and used the last checkpoint.

During training we pad MNIST and Fashion MNIST to 32*32 for convenience and randomly flipped CelebA images.

No other modification was performed.

We only constrained the gradient of the energy function, the energy value itself could in principle be unbounded.

However, we observed that they naturally stabilize so we did not explicitly regularize them.

The annealing sampling schedule is optimized to improve sample quality for CIFAR-10 dataset, and consist of a total of 2700 steps.

For other datasets the shape has less effect on sample quality, see Figure F .1 B for the shape of annealing schedule used.

For the Log likelihood estimation we initialized reverse chain on test images, then sample 10000 intermediate distribution using 10 steps HMC updates each.

Temperature schedule is roughly exponential shaped and the reference distribution is an isotropic Gaussian.

The variance of estimation was generally less than 10% on the log scale.

Due to the high variance of results, and to avoid getting dominated by a single outlier, we report average of the log density instead of log of average density.

We provide more inpainting examples and further demonstrate the mixing during sampling process in Figure E .1.

We also provide more samples for readers to visually judge the quality of our sample generation in Figure E .2, E.3 and E.4.

All samples are randomly selected.

Energy values for CIFAR-10 train, CIFAR-10 test and SVHN datasets for a network trained on CIFAR-10 images.

Note that the network does not over fit to the training set, but just like most deep likelihood model, it assigns lower energy to SVHN images than its own training data.

B. Annealing schedule and a typical energy trace for a sample during Annealed Langevin Sampling.

The energy of the sample is proportional to the temperature, indicating sampling is close to a quasi-static process.

@highlight

Learned energy based model with score matching