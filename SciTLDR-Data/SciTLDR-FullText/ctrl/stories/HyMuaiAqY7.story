Generative Adversarial Networks (GANs) are powerful tools for realistic image generation.

However, a major drawback of GANs is that they are especially hard to train, often requiring large amounts of data and long training time.

In this paper we propose the Deli-Fisher GAN, a GAN that generates photo-realistic images by enforcing structure on the latent generative space using similar approaches in \cite{deligan}. The structure of the latent space we consider in this paper is modeled as a mixture of Gaussians, whose parameters are learned in the training process.

Furthermore, to improve stability and efficiency, we use the Fisher Integral Probability Metric as the divergence measure in our GAN model, instead of the Jensen-Shannon divergence.

We show by experiments that the Deli-Fisher GAN performs better than DCGAN, WGAN, and the Fisher GAN as measured by inception score.

Generative Adversarial Networks (GAN) are powerful unsupervised learning models that have recently achieved great success in learning high-dimensional distributions BID1 ).

In the field of image and vision sciences in particular, GAN models are capable of generating "fake" images that look authentic to human observers.

The basic framework of a GAN model consists of two parts: a generator G = G θ (z) that generates images by translating random input noise z into a particular distribution of interest, and a discriminator D = D p (x) which calculates the probability that an image x is an authentic image as opposed to a generated "fake" image from the generator.

While the generator G and discriminator D can be modeled as any smooth functions, these two components are usually modeled as two neural networks in practical applications.

During the training process, we optimize the generator and the discriminator alternately against each other.

Within each step, we first keep D fixed and optimize G so as to improve its capability of generating images that look real to D. Then, we keep G fixed and train D to improve the discriminator's ability to distinguish real and G-generated images.

The two parts G and D play a two-player game against each other.

At the end of the training, we would be able to have a generator that is capable of generating photo-realistic images.

In mathematical form, a GAN model can be described as an optimization problem, as follows: DISPLAYFORM0 where V (D, G) is the objective function measuring the divergence between the two distributions: the distribution of the real existing data D(x), and the that of the generated data D(G(z)), where x follows the distribution of real images and z follows the distribution of input noise.

Depending on the choice of function V (D, G), different GAN models have been proposed over time (see BID1 , , BID4 ) to increase stability and achieve faster convergence rates.

Ever since the inception of the first GAN models were introduced in BID1 , much improvement has been achieved on the GAN models.

As mentioned in the previous section, the choice of the objective function V (D, G) is crucial to the entire GAN model.

The original GAN model in BID1 optimizes the Jenson-Shannon divergence measure.

This model, however, suffers from slow and unstable training.

Some later work sought to improve GAN performance by utilizing the Earth-Mover Distance ) and the more general f-divergences BID4 ), as well as other possibilities such as the Least Square Objective BID4 ).

Along this line of research, one of the recent notable developments in GANs is the Fisher GAN model proposed by BID4 , which employs the Fisher Integrated Probability Metric (Fisher IPM) to formulate the objective function.

In addition to the developments in divergences used as objective functions in GAN, recent research also focuses on the structure of the latent space for the generator.

In particular, one of the 2017 CVPR papers BID3 introduced Deli-GAN, which uses input noise generated from the mixture of Gaussian distributions.

The paper also argued that this method makes it possible to approximate a huge class of prior data distributions quickly by placing suitable emphasis on noise components, and hence makes training more efficient.

The loss function V (D, G) as shown in FORMULA0 defines how we measure the difference between our learned distribution and the distribution from real images we want to learn.

The divergence measure used in V (D, G) directly controls what the model can achieve through the minimax optimization problem.

Therefore, as shown by recent work, it is important to choose a stable and efficient divergence measure for the loss function.

In the first GAN proposed in BID1 ,the Jensen-Shannon divergence based on KL divergence between two distribution is used, but the model suffers from several problems such as unstable training and slow convergence.

These inherent caveats prompted The WGAN proposed in is more stable and only induces very weak topology (as weak as convergence in distribution), but is known to be costly in computation , BID2 , BID4 ).In this paper, we choose to adopt the Fisher IPM framework proposed by BID4 , which provides stability, efficient computation, and high representation power.

Following the framework developed in BID5 , we define the Integral Probability Metric (IPM).

Let F be the space of all measurable, symmetric, and bounded real functions.

Let X ∈ R d be compact.

Let P and Q be two probability measures on X .

Then the Integral Probability Metric is defined as DISPLAYFORM0 Let P(X ) denote the space of all probability measures on X .

Then d F defines a pseudo-metric over P(X ).

By choosing an appropriate F, we can define a meaningful distance between probability measures.

Now we define the Fisher IPM following the Fisher Discriminative Analysis framework as described in BID4 .

Given two probability measures P, Q ∈ P(X ), the Fisher IPM between P and Q is defined as DISPLAYFORM1 In order to formulate a loss function that is easily computable, we transform the above formula into a constrained format DISPLAYFORM2 so that the problem is better suited for optimization, as we will see in the following sections.

Most GAN models introduced in previous work BID1 , , BID4 ) make use of random noise generated from a uniform distribution or a Gaussian distribution in their latent space for the input to the generator.

These choices of using overly simplistic distributions, however, are not well justified.

Since the data we train the GAN upon is often diverse with many varying classes of images.

choosing one uniform or Gaussian distribution to generate the random noise input may fail to represent the features in the latent space.

We believe that a good choice of probability distribution for the latent noise will be able to translate into better features or structures in the generated image.

An idea of using mixed Gaussian distribution in the latent space was proposed in BID3 , in which the authors changed distribution of the random noise input from a singular uniform/Gaussian distributions to a mixture of Gaussians, and incorporated the GAN architecture from the DCGAN model described in BID6 .

During the training process, the parameters of the mixed Gaussian distribution (means and variances) are learned in each epoch.

Once the training is complete, the Deli-GAN generates images using the mixed Gaussian learned from training process.

Thus, we incorporate this idea in our paper, and generalize the distribution of the latent space to general mixture distributions: DISPLAYFORM0 where D θi are all Gaussian distributions, then θ i = (µ i , σ i ) represent the means and standard deviations of these Gaussians.

Using the mixture input random noise, we proceed to build the GAN model with the Fisher IPM we have described in the previous section.

The following sections will discuss in detail of the loss function and algorithms implemented.

By our discussion above, we reformulate the Deli-Fisher GAN model into the following optimization problem: DISPLAYFORM0 where P r is the distribution of the real images and P DISPLAYFORM1 g is the distribution of the i th component of latent input noise, as a multimodal distribution.

g .

In a simple case, if the P (i) g 's are independently and identically distributed, and α i only depend on their means µ i and variances σ i , i.e. α i = α i (µ i , σ i ), then the empirical formulation of (2) can be written as DISPLAYFORM0 Here, N , M are our sample sizes for the discriminator and the generator respectively, and C is a constant controlling the size of σ.

λ represents the Lagrange multiplier for optimization, while ρ and β are penalty weights for the L 2 -regularity ofΩ and σ, respectively.

i are random noises that provides diversity to the latent space.

i are sampled from the normalized P (i) g .

The parameters for our structured noise input are in turn updated during training process, as in the case with BID3 .

Using the standard stochastic gradient descent(SGD) algorithm ADAM, over all sets of parameters, we compute the updates of the respective variables by optimizing the loss functions described in the previous section with the following procedure:Input: ρ penalty weight, η, η learning rates, n c number of iterations for training the critic, N batch size Initialize p, θ, λ Initialize µ i , σ i , η while θ not converging do for j = 1 to n c do Sample a minibatch DISPLAYFORM0 Algorithm 1: Deli-Fisher GAN

To evaluate the quality of the images generated by our GAN, we use the Inception Score as defined in BID7 , an automated measure aiming to simulate human judgment of quality of the images generated.

This measure aims to minimize the entropy for the conditional label distribution p(y|x) to ensure consistency between the generated images and given data, and maximize the entropy of the marginal p(y|x = G(z))dz to guarantee the diversity of the images generated by the network.

In view of these two considerations, the proposed metric can be written as DISPLAYFORM0 where D KL (p q) denotes the Kullback-Leibler divergence between two distributions p and q. An alternative measure involving the exponent of inception score has been proposed in BID3 ; for our experiments, we will stick to the original formulation as proposed in BID7 .The inception score we used in all experiments below is calculated by the python script posted by OpenAI at https://github.com/openai/improved-gan/tree/master/ inception_score.

As a baseline for subsequent comparison, we have replicated the experiments of previous GAN architectures.

We have successfully replicated the results for Deep Convolutional Generative Adversarial Networks (DC-GAN) in BID6 , Wasserstein GAN in , and Fisher GAN in BID4 , all using the data set CIFAR-10.

TAB0 are two tables that show the results of our experimental replication and the means and variances of their respective inception scores.

we used cropped images of size 32 × 32 so that the dense layer of our neural networks does not become too large.

For each training, We generated 50,000 fake images and used these images to calculate the inception score.

Each the training session consists of 200 epochs.

In each session, we applied generated corresponding output.

Then we apply Deli-Fisher GAN to the same data set and compare the result with Fisher-GAN.

In the Deli-Fisher GAN, we set hyper-parameters as 0 and initialized parameters for the input distribution (µ i , σ i and η).

We executed same number of epochs in the training session.

During the training session, θ, µ, σ and η were learned by Stochastic Gradient Descent with ADAM optimizer.

After we have learned the parameters of the model, we generated another 50,000 images to make comparison with those generated by Fisher-GAN.At the same time, we have also tuned different parameters in each model generation to fake sample production work-flow.

These parameters include the number of epochs, the penalty coefficient, etc.

We have also made use of the inception score described above to compare the images we've generated with the ones in the original data distribution.

All the experiments are done on GeForce GTX 1080Ti GPU, and we have observed that most of the GAN trainings involved in our experiments take around 30 minutes.

One notable exception, however, lies in WGAN, since the weight-clipping procedures involved in WGAN requires a lot of computation and accounts for the extra time needed in experiments.

Moreover, while repeating the experiments of different GANs, we noticed that the performances of DCGAN were highly unstable and unsatisfactory, as DCGAN yielded varying unsatisfactory inception scores at the range of 2 to 3 in our runs and stopped parameter updating even when the images are still blurred.

These observations confirm the conclusions in and BID4 .

Using suitable parameters located through fine-tuning, the Deli-Fisher GAN produces better images than those produced by the current optimal Fisher GAN, as measured by the inception score.

For comparison, the respective inception scores in experiments over the CIFAR-10 dataset are listed in Table 3 .

As demonstrated by the tables, experiments generated images of good qualities.

One such sample is shown in FIG2 .

Compared with previous GANs, we can see notable improvements in the images generated, by qualitative and quantitative observation.

These outputs therefore suggest that a better representation of the random noise input does indeed capture more features of the latent space and those of the images the model is trained upon, and these features, in turn, augment the authenticity of the images that the Deli-Fisher GAN model produces.

In sum, the Deli-Fisher GAN presented in our paper is capable of generating better images than the DC-GAN, the WGAN, and the Fisher-GAN are, with notable improvements on the quality of images as measured by inception scores.

Additionally, the model proposed in our paper is still open to improvement such as adding regularization terms to the objective function as those employed in the experiments of BID4 .As a further step, we are working on developing more sophisticated structures for the latent space that is specific tailored to different tasks.

We believe, by enforcing some properties on the latent space, e.g. symmetries or geometric characteristics, we would be able to gain some control of the features on the generated images.

<|TLDR|>

@highlight

This paper proposes a new Generative Adversarial Network that is more stable, more efficient, and produces better images than those of status-quo 

@highlight

This paper combines Fisher-GAN and Deli-GAN

@highlight

This paper combines Deli-GAN, which has a mixture prior distribution in latent space, and Fisher GAN, which uses Fisher IPM instead of JSD as an objective.