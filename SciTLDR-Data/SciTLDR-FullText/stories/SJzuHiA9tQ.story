Generative Adversarial Networks (GANs) have proven to be a powerful framework for learning to draw samples from complex distributions.

However, GANs are also notoriously difficult to train, with mode collapse and oscillations a common problem.

We hypothesize that this is at least in part due to the evolution of the generator distribution and the catastrophic forgetting tendency of neural networks, which leads to the discriminator losing the ability to remember synthesized samples from previous instantiations of the generator.

Recognizing this, our contributions are twofold.

First, we show that GAN training makes for a more interesting and realistic benchmark for continual learning methods evaluation than some of the more canonical datasets.

Second, we propose leveraging continual learning techniques to augment the discriminator, preserving its ability to recognize previous generator samples.

We show that the resulting methods add only a light amount of computation, involve minimal changes to the model, and result in better overall performance on the examined image and text generation tasks.

Generative Adversarial Networks BID6 (GANs) are a popular framework for modeling draws from complex distributions, demonstrating success in a wide variety of settings, for example image synthesis BID14 and language modeling .

In the GAN setup, two agents, the discriminator and the generator (each usually a neural network), are pitted against each other.

The generator learns a mapping from an easy-to-sample latent space to a distribution in the data space, which ideally matches the real data's distribution.

At the same time, the discriminator aims to distinguish the generator's synthesized samples from the real data samples.

When trained successfully, GANs yield impressive results; in the image domain for example, synthesized images from GAN models are significantly sharper and more realistic than those of other classes of models BID16 .

On the other hand, GAN training can be notoriously finicky.

One particularly well-known and common failure mode is mode collapse BID0 BID35 : instead of producing samples sufficiently representing the true data distribution, the generator maps the entire latent space to a limited subset of the real data space.

When mode collapse occurs, the generator does not "converge," in the conventional sense, to a stationary distribution.

Rather, because the discriminator can easily learn to recognize a modecollapsed set of samples and the generator is optimized to avoid the discriminator's detection, the two end up playing a never-ending game of cat and mouse: the generator meanders towards regions in the data space the discriminator thinks are real (likely near where the real data lie) while the discriminator chases after it.

Interestingly though, if generated samples are plotted through time (as in FIG0 ), it appears that the generator can revisit previously collapsed modes.

At first, this may seem odd.

The discriminator was ostensibly trained to recognize that mode in a previous iteration and did so well enough to push the generator away from generating those samples.

Why has the discriminator seemingly lost this ability?We conjecture that this oscillation phenomenon is enabled by catastrophic forgetting BID20 BID30 : neural networks have a well-known tendency to forget how to complete old tasks while learning new ones.

In most GAN models, the discriminator is a binary classifier, with the two classes being the real data and the generator's outputs.

Implicit to the training of a standard classifier is the assumption that the data are drawn independently and identically distributed (i.i.d.) .

Importantly, this assumption does not hold true in GANs: the distribution of the generator class (and thus the discriminator's training data) evolves over time.

Moreover, these changes in the generator's distribution are adversarial, designed specifically to deteriorate discriminator performance on the fake class as much as possible.

Thus, the alternating training procedure of GANs in actuality corresponds to the discriminator learning tasks sequentially, where each task corresponds to recognizing samples from the generator at that particular point in time.

Without any measures to prevent catastrophic forgetting, the discriminator's ability to recognize fake samples from previous iterations will be clobbered by subsequent gradient updates, allowing a mode-collapsed generator to revisit old modes if training runs long enough.

Given this tendency, a collapsed generator can wander indefinitely without ever learning the true distribution.

With this perspective in mind, we cast training the GAN discriminator as a continual learning problem, leading to two main contributions.

(i) While developing systems that learn tasks in a sequential manner without suffering from catastrophic forgetting has become a popular direction of research, current benchmarks have recently come under scrutiny as being unrepresentative to the fundamental challenges of continual learning BID3 .

We argue that GAN training is a more realistic setting, and one that current methods tend to fail on.(ii) Such a reframing of the GAN problem allows us to leverage relevant methods to better match the dynamics of training the min-max objective.

In particular, we build upon the recently proposed elastic weight consolidation BID15 and intelligent synapses BID39 .

By preserving the discriminator's ability to identify previous generator samples, this memory prevents the generator from simply revisiting past distributions.

Adapting the GAN training procedure to account for catastrophic forgetting provides an improvement in GAN performance for little computational cost and without the need to train additional networks.

Experiments on CelebA and CIFAR10 image generation and COCO Captions text generation show discriminator continual learning leads to better generations.

Consider distribution p real (x), from which we have data samples D real .

Seeking a mechanism to draw samples from this distribution, we learn a mapping from an easy-to-sample latent distribution p(z) to a data distribution p gen (x), which we want to match p real (x).

This mapping is parameterized as a neural network G φ (z) with parameters φ, termed the generator.

The synthesized data are drawn x = G φ (z), with z ∼ p(z).

The form of p gen (x) is not explicitly assumed or learned; rather, we learn to draw samples from p gen (x).To provide feedback to G φ (z), we simultaneously learn a binary classifier that aims to distinguish synthesized samples D gen drawn from p gen (x) from the true samples D real .

We also parameterize this classifier as a neural network D θ (x) ∈ [0, 1] with parameters θ, with D θ (x) termed the discriminator.

By incentivizing the generator to fool the discriminator into thinking its generations are actually from the true data, we hope to learn G φ (z) such that p gen (x) approaches p real (x).These two opposing goals for the generator and discriminator are usually formulated as the following min-max objective: DISPLAYFORM0 At each iteration t, we sample from p gen (x), yielding generated data D gen t .

These generated samples, along with samples from D real , are then passed to the discriminator.

A gradient descent optimizer nudges θ so that the discriminator takes a step towards maximizing L GAN (θ, φ).

Parameters φ are updated similarly, but to minimize L GAN (θ, φ).

These updates to θ and φ take place in an alternating fashion.

The expectations are approximated using samples from the respective distributions, and therefore learning only requires observed samples D real and samples from p gen (x).The updates to G φ (z) mean that p gen (x) changes as a function of t, perhaps substantially.

(x) .

Because of the catastrophic forgetting effect of neural networks, the ability of D θ (x) to recognize these previous distributions is eventually lost in the pursuit of maximizing L GAN (θ, φ) with respect to only D gen t .

This opens the possibility that the generator goes back to generating samples the discriminator had previously learned (and then forgot) to recognize, leading to unstable mode-collapsed oscillations that hamper GAN training (as in FIG0 ).

Recognizing this problem, we propose that the discriminator should be trained with the temporal component of p gen (x) in mind.

3.1 CLASSIC CONTINUAL LEARNING Catastrophic forgetting has long been known to be a problem with neural networks trained on a series of tasks BID20 BID30 .

While there are many approaches to addressing catastrophic forgetting, here we primarily focus on elastic weight consolidation (EWC) and intelligent synapses (IS).

These are meant to illustrate the potential of catastrophic forgetting mitigation to improve GAN learning, with the expectation that this opens up the possibility of other such methods to significantly improve GAN training, at low additional computational cost.

To derive the EWC loss, BID15 frames training a model as finding the most probable values of the parameters θ given the data D. For two tasks, the data are assumed partitioned into independent sets according to the task, and the posterior for Task 1 is approximated as a Gaussian with mean centered on the optimal parameters for Task 1 θ * 1 and diagonal precision given by the diagonal of the Fisher information matrix F 1 at θ * 1 .

This gives the EWC loss the following form: DISPLAYFORM0 where L 2 (θ) = log p(D 2 |θ) is the loss for Task 2 individually, λ is a hyperparameter representing the importance of Task 1 relative to Task 2, DISPLAYFORM1 2 , i is the parameter index, and L(θ) is the new loss to optimize while learning Task 2.

Intuitively, the EWC loss prevents the model from straying too far away from the parameters important for Task 1 while leaving less crucial parameters free to model Task 2.

Subsequent tasks result in additional L EWC (θ) terms added to the loss for each previous task.

By protecting the parameters deemed important for prior tasks, EWC as a regularization term allows a single neural network (assuming sufficient parameters and capacity) to learn new tasks in a sequential fashion, without forgetting how to perform previous tasks.

While EWC makes a point estimate of how essential each parameter is at the conclusion of a task, IS BID39 protects the parameters according to their importance along the task's entire training trajectory.

Termed synapses, each parameter θ i of the neural network is awarded an importance measure ω 1,i based on how much it reduced the loss while learning Task 1.

Given a loss gradient g(t) = ∇ θ L(θ)| θ=θt at time t, the total change in loss during the training of Task 1 then is the sum of differential changes in loss over the training trajectory.

With the assumption that parameters θ are independent, we have:where θ = dθ dt and (t 0 , t 1 ) are the start and finish of Task 1, respectively.

Note the added negative sign, as importance is associated with parameters that decrease the loss.

The importance measure ω 1,i can now be used to introduce a regularization term that protects parameters important for Task 1 from large parameter updates, just as the Fisher information matrix diagonal terms F 1,i were used in EWC.

This results in an IS loss very reminiscent in form 1 : DISPLAYFORM0 3.2 GAN CONTINUAL LEARNING The traditional continual learning methods are designed for certain canonical benchmarks, commonly consisting of a small number of clearly defined tasks (e.g., classification datasets in sequence).

In GANs, the discriminator is trained on dataset DISPLAYFORM1 However, because of the evolution of the generator, the distribution p gen (x) from which D gen t comes changes over time.

This violates the i.i.d.

assumption of the order in which we present the discriminator data.

As such, we argue that different instances in time of the generator should be viewed as separate tasks.

Specifically, in the parlance of continual learning, the training data are to be regarded as DISPLAYFORM2 Thus motivated, we would like to apply continual learning methods to the discriminator, but doing so is not straightforward for the following reasons:• Definition of a task: EWC and IS were originally proposed for discrete, well-defined tasks.

For example, BID15 applied EWC to a DQN BID25 learning to play ten Atari games sequentially, with each game being a clear, independent task.

For GAN, there is no such precise definition as to what constitutes a "task," and as discriminators are not typically trained to convergence at every iteration, it is also unclear how long a task should be.• Computational memory: While Equations 2 and 4 are for two tasks, they can be extended to K tasks by adding a term L EWC or L IS for each of the K − 1 prior tasks.

As each term L EWC or L IS requires saving both a historical reference term θ * k and either F k or ω k (all of which are the same size as the model parameters θ) for each task k, employing these techniques naively quickly becomes impractical for bigger models when K gets large, especially if K is set to the number of training iterations T .•

Continual not learning: Early iterations of the discriminator are likely to be non-optimal, and without a forgetting mechanism, EWC and IS may forever lock the discriminator to a poor initialization.

Additionally, the unconstrained addition of a large number of terms L EWC or L IS will cause the continual learning regularization term to grow unbounded, which can disincentivize any further changes in θ.

To address these issues, we build upon the aforementioned continual learning techniques, and propose several changes.

Number of tasks as a rate: We choose the total number of tasks K as a function of a constant rate α, which denotes the number of iterations before the conclusion of a task, as opposed to arbitrarily dividing the GAN training iterations into some set number of segments.

Given T training iterations, this means a rate α yields K = T α tasks.

Online Memory: Seeking a way to avoid storing extra θ * k , F k , or ω k , we observe that the sum of two or more quadratic forms is another quadratic, which gives the classifier loss with continual learning the following form for the (k + 1) th task: DISPLAYFORM3 DISPLAYFORM4 κ,i , and Q κ,i is either F κ,i or ω κ,i , depending on the method.

We name models with EWC and IS augmentations EWC-GAN and IS-GAN, respectively.1 BID39 DISPLAYFORM5 , where ∆1,i = θ1,i − θ0,i and ξ is a small number for numerical stability.

We however found that the inclusion of (∆1,i) 2 can lead to the loss exploding and then collapsing as the number of tasks increases and so omit it.

We also change the hyperparameter c into Under review as a conference paper at ICLR 2019Controlled forgetting: To provide a mechanism for forgetting earlier non-optimal versions of the discriminator and to keep L CL bounded, we add a discount factor γ: DISPLAYFORM6 Together, α and γ determine how far into the past the discriminator remembers previous generator distributions, and λ controls how important memory is relative to the discriminator loss.

Note, the terms S k and P k can be updated every α steps in an online fashion: DISPLAYFORM7 This allows the EWC or IS loss to be applied without necessitating storing either Q k or θ * k for every task k, which would quickly become too costly to be practical.

Only a single variable to store a running average is required for each of S k and P k , making this method space efficient.

Augmenting the discriminator with the continual learning loss, the GAN objective becomes: DISPLAYFORM8 Note that the training of the generator remains the same; full algorithms are in Appendix A. Here we have shown two methods to mitigate catastrophic forgetting for the original GAN; however, the proposed framework is applicable to almost all of the wide range of GAN setups.

Continual learning in GANs There has been previous work investigating continual learning within the context of GANs.

Improved GAN BID32 introduced historical averaging, which regularizes the model with a running average of parameters of the most recent iterations.

Simulated+Unsupervised training BID34 proposed replacing half of each minibatch with previous generator samples during training of the discriminator, as a generated sample at any point in time should always be considered fake.

However, such an approach necessitates a historical buffer of samples and halves the number of current samples that can be considered.

Continual Learning GAN BID33 applied EWC to GAN, as we have, but used it in the context of the class-conditioned generator that learns classes sequentially, as opposed to all at once, as we propose.

BID36 independently reached a similar conclusion on catastrophic forgetting in GANs, but focused on gradient penalties and momentum on toy problems.

The heart of continual learning is distilling a network's knowledge through time into a single network, a temporal version of the ensemble described in BID10 .

There have been several proposed models utilizing multiple generators BID4 or multiple discriminators BID2 BID27 , while Bayesian GAN BID31 considered distributions on the parameters of both networks, but these all do not consider time as the source of the ensemble.

Unrolled GAN BID22 ) considered multiple discriminators "unrolled" through time, which is similar to our method, as the continual learning losses also utilize historical instances of discriminators.

However, both EWC-GAN and IS-GAN preserve the important parameters for prior discriminator performance, as opposed to requiring backpropagation of generator samples through multiple networks, making them easier to implement and train.

GAN convergence While GAN convergence is not the focus of this paper, convergence does similarly avoid mode collapse, and there are a number of works on the topic BID9 BID37 BID26 BID21 .

From the perspective of BID9 , EWC or IS regularization in GAN can be viewed as achieving convergence by slowing the discriminator, but per parameter, as opposed to a slower global learning rate.

5.1 DISCRIMINATOR CATASTROPHIC FORGETTING While FIG0 implies catastrophic forgetting in a GAN discriminator, we can show this concretely.

To do so, we first train a DCGAN on the MNIST dataset.

Since the generator is capable of generating an arbitrary number of samples at any point, we can randomly draw 70000 samples to comprise a new, "fake MNIST" dataset at any time.

By doing this at regular intervals, we create datasets {D Having previously generated a series of datasets during the training of a DCGAN, we now reinitialize the discriminator and train to convergence on each D after fine-tuning on D gen t ; this is unsurprising, as p gen (x) has evolved specifically to deteriorate discriminator performance.

While there is still a dropoff with EWC, forgetting is less severe.

While the training outlined above is not what is typical for GAN, we choose this set-up as it closely mirrors the continual learning literature.

With recent criticisms of some common continual learning benchmarks as either being too easy or missing the point of continual learning BID3 , we propose GAN as a new benchmark providing a more realistic setting.

From FIG3 , it is clear that while EWC certainly helps, there is still much room to improve with new continual learning methods.

However, the merits of GAN as a continual learning benchmark go beyond difficulty.

While it is unclear why one would ever use a single model to classify successive random permutations of MNIST (Goodfellow et al., 2013) , many real-world settings exist where the data distribution is slowly evolving.

For such models, we would like to be able to update the deployed model without forgetting previously learned performance, especially when data collection is expensive and thus done in bulk sometime before deployment.

For example, autonomous vehicles BID12 will eventually encounter unseen car models or obstacles, and automated screening systems at airport checkpoints BID18 will have to deal with evolving bags, passenger belongings, and threats.

In both cases, sustained effectiveness requires a way to appropriately and efficiently update the models for new data, or risk obsolescence leading to dangerous blindspots.

Many machine learning datasets represent singe-time snapshots of the data distribution, and current continual learning benchmarks fail to capture the slow drift of the real-world data.

The evolution of GAN synthesized samples represents an opportunity to generate an unlimited number of smoothly evolving datasets for such experiments.

We note that while the setup used here is for binary real/fake classification, one could also conceivably use a conditional GAN BID23 to generate an evolving multi-class classification dataset.

We leave this exploration for future work.

We show results on a toy dataset consisting of a mixture of eight Gaussians, as in the example in FIG0 .

Following the setup of BID22 , the real data are evenly distributed among eight 2-dimensional Gaussian distributions arranged in a circle of radius 2, each with covariance 0.02I (see Figure 4) .

We evaluate our model with Inception Score (ICP) BID32 , which gives a rough measure of diversity and quality of samples; higher scores imply better performance, with the true data resulting in a score of around 7.870.

For this simple dataset, since we know the true data distribution, we also calculate the symmetric Kullback-Leibler divergence (Sym-KL); lower scores mean the generated samples are closer to the true data.

We show computation time, measured in numbers of training iterations per second (Iter/s), averaged over the full training of a model on a single Nvidia Titan X (Pascal) GPU.

Each model was run 10 times, with the mean and standard deviation of each performance metric at the end of 25K iterations reported in TAB1 .

The performance of EWC-GAN and IS-GAN were evaluated for a number of hyperparameter settings.

We compare our results against a vanilla GAN (Goodfellow et al., 2014) , as well as a state-ofthe-art GAN with spectral normalization (SN) BID24 applied to the discriminator.

As spectral normalization augments the discriminator loss in a way different from continual learning, we can combine the two methods; this variant is also shown.

Note that a discounted version of discriminator historical averaging BID32 can be recovered from the EWC and IS losses if the task rate α = 1 and Q k,i = 1 for all i and k, a poor approximation to both the Fisher information matrix diagonal and importance measure.

If we also set the historical reference termθ * k and the discount factor γ to zero, then the EWC and IS losses become 2 weight regularization.

These two special cases are also included for comparison.

We observe that augmenting GAN models with EWC and IS consistently results in generators that better match the true distribution, both qualitatively and quantitatively, for a wide range of hyperparameter settings.

EWC-GAN and IS-GAN result in a better ICP and FID than 2 weight regularization and discounted historical averaging, showing the value of prioritizing protecting important parameters, rather than all parameters equally.

EWC-GAN and IS-GAN also outperform a stateof-the-art method in SN-GAN.

In terms of training time, updating the EWC loss requires forward propagating a new minibatch through the discriminator and updating S and P , but even if this is done at every step (α = 1), the resulting algorithm is only slightly slower than SN-GAN.

Moreover, doing so is unnecessary, as higher values of α also provide strong performance for a much smaller time penalty.

Combining EWC with SN-GAN leads to even better results, showing that the two methods can complement each other.

IS-GAN can also be successfully combined with SN-GAN, but it is slower than EWC-GAN as it requires tracking the trajectory of parameters at each step.

Sample generation evolution over time is shown in Figure 4 of Appendix C.

Since EWC-GAN achieves similar performance to IS-GAN but at less computational expense, we focus on the former for experiments on two image datasets, CelebA and CIFAR-10.

Our EWC-GAN implementation is straightforward to add to any GAN model, so we augment various popular implementations.

Comparisons are made with the TTUR BID9 variants 2 of DCGAN and WGAN-GP BID7 , as well as an implementation 3 of a spectral normalized BID24 ) DCGAN (SN-DCGAN) .

Without modifying the learning rate or model architecture, we show results with and without the EWC loss term added to the discriminator for each.

Performance is quantified with the Fréchet Inception Distance (FID) BID9 for both datasets.

Since labels are available for CIFAR-10, we also report ICP for that dataset.

Best values are reported in TAB2 , with samples in Appendix C. In each model, we see improvement in both FID and ICP from the addition of EWC to the discriminator.

We also consider the text generation on the MS COCO Captions dataset BID1 , with the pre-processing in BID8 .

Quality of generated sentences is evaluated by BLEU score BID28 .

Since BLEU-b measures the overlap of b consecutive words between the generated sentences and ground-truth references, higher BLEU scores indicate better fluency.

Self BLEU uses the generated sentences themselves as references; lower values indicate higher diversity.

We apply EWC and IS to textGAN , a recently proposed model for text generation in which the discriminator uses feature matching to stabilize training.

This model's results (labeled "EWC" and "IS") are compared to a Maximum Likelihood Estimation (MLE) baseline, as well as several state-of-the-art methods: SeqGAN BID38 , RankGAN BID19 , GSGAN BID13 and LeakGAN (Guo et al., 2018) .

Our variants of textGAN outperforms the vanilla textGAN for all BLEU scores (see TAB3 ), indicating the effectiveness of addressing the forgetting issue for GAN training in text generation.

EWC/IS + textGAN also demonstrate a significant improvement compared with other methods, especially on BLEU-2 and 3.

Though our variants lag slightly behind LeakGAN on BLEU-4 and 5, their self BLEU scores TAB4 indicate it generates more diverse sentences.

Sample sentence generations can be found in Appendix C.

We observe that the alternating training procedure of GAN models results in a continual learning problem for the discriminator, and training on only the most recent generations leads to consequences unaccounted for by most models.

As such, we propose augmenting the GAN training objective with a continual learning regularization term for the discriminator to prevent its parameters from moving too far away from values that were important for recognizing synthesized samples from previous training iterations.

Since the original EWC and IS losses were proposed for discrete tasks, we adapt them to the GAN setting.

Our implementation is simple to add to almost any variation of GAN learning, and we do so for a number of popular models, showing a gain in ICP and FID for CelebA and CIFAR-10, as well as BLEU scores for COCO Captions.

More importantly, we demonstrate that GAN and continual learning, two popular fields studied independently of each other, have the potential to benefit each other, as new continual learning methods stand to benefit GAN training, and GAN generated datasets provide new testing grounds for continual learning.

To produce a smoothly evolving series of datasets for continual learning, we train a DCGAN on MNIST and generate an entire "fake" dataset of 70K samples every 50 training iterations of the DC-GAN generator.

We propose learning each of these generated datasets as individual tasks for continual learning.

Selected samples are shown in Figure 3 from the datasets D gen t for t ∈ {5, 10, 15, 20}, each generated from the same 100 samples of z for all t. Note that we actually trained a conditional DCGAN, meaning we also have the labels for each generated image.

For experiments in FIG3 , we focused on the real versus fake task to demonstrate catastrophic forgetting in a GAN discriminator and thus ignored the labels, but future experiments can incorporate such information.

Figure 3: Image samples from generated "fake MNIST" datasets C EXAMPLES OF GENERATED SAMPLES Sample generations are plotted during training at 5000 step intervals in Figure 4 .

While vanilla GAN occasionally recovers the true distribution, more often than not, the generator collapses and then bounces around.

Spectral Normalized GAN converges to the true distribution quickly in most runs, but it mode collapses and exhibits the same behavior as GAN in others.

EWC-GAN consistently diffuses to all modes, tending to find the true distribution sooner with lower α.

We omit IS-GAN, as it performs similarly to EWC-GAN.

Figure 4: Each row shows the evolution of generator samples at 5000 training step intervals for GAN, SN-GAN, and EWC-GAN for two α values.

The proposed EWC-GAN models have hyperparameters matching the corresponding α in TAB1 .

Each frame shows 10000 samples drawn from the true eight Gaussians mixture (red) and 10000 generator samples (blue).

We also show the generated image samples for CIFAR 10 and CelebA in Figure 5 , and generated text samples for MS COCO Captions in Table 5 .(a) CIFAR 10 (b) CelebA Figure 5 : Generated image samples from random draws of EWC+GANs.

Table 5 : Sample sentence generations from EWC + textGAN a couple of people are standing by some zebras in the background the view of some benches near a gas station a brown motorcycle standing next to a red fence a bath room with a broken tank on the floor red passenger train parked under a bridge near a river some snow on the beach that is surrounded by a truck a cake that has been perform in the background for takeoff a view of a city street surrounded by trees two giraffes walking around a field during the day crowd of people lined up on motorcycles two yellow sheep with a baby dog in front of other sheep an intersection sits in front of a crowd of people a red double decker bus driving down the street corner an automobile driver stands in the middle of a snowy park five people at a kitchen setting with a woman there are some planes at the takeoff station a passenger airplane flying in the sky over a cloudy sky three aircraft loaded into an airport with a stop light there is an animal walking in the water an older boy with wine glasses in an office two old jets are in the middle of london three motorcycles parked in the shade of a crowd group of yellow school buses parked on an intersection a person laying on a sidewalk next to a sidewalk talking on a cell phone a chef is preparing food with a sink and stainless steel appliances

@highlight

Generative Adversarial Network Training is a Continual Learning Problem.