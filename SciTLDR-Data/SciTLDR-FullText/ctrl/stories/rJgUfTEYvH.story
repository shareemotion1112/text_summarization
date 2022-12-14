Generative models that can model and predict sequences of future events can, in principle, learn to capture complex real-world phenomena, such as physical interactions.

However, a central challenge in video prediction is that the future is highly uncertain: a sequence of past observations of events can imply many possible futures.

Although a number of recent works have studied probabilistic models that can represent uncertain futures, such models are either extremely expensive computationally as in the case of pixel-level autoregressive models, or do not directly optimize the likelihood of the data.

To our knowledge, our work is the first to propose multi-frame video prediction with normalizing flows, which allows for direct optimization of the data likelihood, and produces high-quality stochastic predictions.

We describe an approach for modeling the latent space dynamics, and demonstrate that flow-based generative models offer a viable and competitive approach to generative modeling of video.

Exponential progress in the capabilities of computational hardware, paired with a relentless effort towards greater insights and better methods, has pushed the field of machine learning from relative obscurity into the mainstream.

Progress in the field has translated to improvements in various capabilities, such as classification of images (Krizhevsky et al., 2012) , machine translation (Vaswani et al., 2017) and super-human game-playing agents (Mnih et al., 2013; Silver et al., 2017) , among others.

However, the application of machine learning technology has been largely constrained to situations where large amounts of supervision is available, such as in image classification or machine translation, or where highly accurate simulations of the environment are available to the learning agent, such as in game-playing agents.

An appealing alternative to supervised learning is to utilize large unlabeled datasets, combined with predictive generative models.

In order for a complex generative model to be able to effectively predict future events, it must build up an internal representation of the world.

For example, a predictive generative model that can predict future frames in a video would need to model complex real-world phenomena, such as physical interactions.

This provides an appealing mechanism for building models that have a rich understanding of the physical world, without any labeled examples.

Videos of real-world interactions are plentiful and readily available, and a large generative model can be trained on large unlabeled datasets containing many video sequences, thereby learning about a wide range of real-world phenoma.

Such a model could be useful for learning representations for further downstream tasks (Mathieu et al., 2016) , or could even be used directly in applications where predicting the future enables effective decision making and control, such as robotics (Finn et al., 2016) .

A central challenge in video prediction is that the future is highly uncertain: a short sequence of observations of the present can imply many possible futures.

Although a number of recent works have studied probabilistic models that can represent uncertain futures, such models are either extremely expensive computationally (as in the case of pixel-level autoregressive models), or do not directly optimize the likelihood of the data.

In this paper, we study the problem of stochastic prediction, focusing specifically on the case of conditional video prediction: synthesizing raw RGB video frames conditioned on a short context of past observations (Ranzato et al., 2014; Srivastava et al., 2015; Vondrick et al., 2015; Xingjian et al., 2015; Boots et al., 2014) .

Specifically, we propose a new class of video prediction models that can provide exact likelihoods, generate diverse stochastic futures, and accurately synthesize realistic and high-quality video frames.

The main idea behind our approach is to extend flow-based generative models (Dinh et al., 2014; into the setting of conditional video prediction.

To our knowledge, flow-based models have been applied only to generation of non-temporal data, such as images (Kingma & Dhariwal, 2018) , and to audio sequences (Prenger et al., 2018) .

Conditional generation of videos presents its own unique challenges: the high dimensionality of video sequences makes them difficult to model as individual datapoints.

Instead, we learn a latent dynamical system model that predicts future values of the flow model's latent state.

This induces Markovian dynamics on the latent state of the system, replacing the standard unconditional prior distribution.

We further describe a practically applicable architecture for flow-based video prediction models, inspired by the Glow model for image generation (Kingma & Dhariwal, 2018) , which we call VideoFlow.

Our empirical results show that VideoFlow achieves results that are competitive with the state-ofthe-art in stochastic video prediction on the action-free BAIR dataset, with quantitative results that rival the best VAE-based models.

VideoFlow also produces excellent qualitative results, and avoids many of the common artifacts of models that use pixel-level mean-squared-error for training (e.g., blurry predictions), without the challenges associated with training adversarial models.

Compared to models based on pixel-level autoregressive prediction, VideoFlow achieves substantially faster test-time image synthesis 1 , making it much more practical for applications that require real-time prediction, such as robotic control .

Finally, since VideoFlow directly optimizes the likelihood of training videos, without relying on a variational lower bound, we can evaluate its performance directly in terms of likelihood values.

Early work on prediction of future video frames focused on deterministic predictive models (Ranzato et al., 2014; Srivastava et al., 2015; Vondrick et al., 2015; Xingjian et al., 2015; Boots et al., 2014) .

Much of this research on deterministic models focused on architectural changes, such as predicting high-level structure Villegas et al. (2017b) , incorporating pixel transformations (Finn et al., 2016; De Brabandere et al., 2016; Liu et al., 2017) and predictive coding architectures (Lotter et al., 2017) , as well as different generation objectives (Mathieu et al., 2016; Vondrick & Torralba, 2017; Walker et al., 2015) and disentangling representations (Villegas et al., 2017a; Denton & Birodkar, 2017) .

With models that can successfully model many deterministic environments, the next key challenge is to address stochastic environments by building models that can effectively reason over uncertain futures.

Real-world videos are always somewhat stochastic, either due to events that are inherently random, or events that are caused by unobserved or partially observable factors, such as off-screen events, humans and animals with unknown intentions, and objects with unknown physical properties.

In such cases, since deterministic models can only generate one future, these models either disregard potential futures or produce blurry predictions that are the superposition or averages of possible futures.

A variety of methods have sought to overcome this challenge by incorporating stochasticity, via three types of approaches: models based on variational auto-encoders (VAEs) (Kingma & Welling, 2013; Rezende et al., 2014) , generative adversarial networks (Goodfellow et al., 2014) , and autoregressive models (Hochreiter & Schmidhuber, 1997; Graves, 2013; van den Oord et al., 2016b; c; Van Den Oord et al., 2016) .

Among these models, techniques based on variational autoencoders which optimize an evidence lower bound on the log-likelihood have been explored most widely Denton & Fergus, 2018; Lee et al., 2018; Xue et al., 2016; Li et al., 2018) .

To our knowledge, the only prior class of video prediction models that directly maximize the log-likelihood of the data are auto-regressive models (Hochreiter & Schmidhuber, 1997; Graves, 2013; van den Oord et al., 2016b; c; Van Den Oord et al., 2016) , that generate the video one pixel at a time .

However, synthesis with such models is typically inherently sequential, making synthesis substantially inefficient on modern parallel hardware.

Prior work has aimed to speed up training and synthesis with such auto-regressive models (Reed et al., 2017; Ramachandran et al., 2017) .

However, show that the predictions from these models are sharp but noisy and that the proposed VAE model produces substantially better predictions, especially for longer horizons.

In contrast to autoregressive models, we find that our proposed method exhibits faster sampling, while still directly optimizing the log-likelihood and producing high-quality long-term predictions.

. . .

Figure 1: Left: Multi-scale prior The flow model uses a multi-scale architecture using several levels of stochastic variables.

Right: Autoregressive latent-dynamic prior The input at each timestep xt is encoded into multiple levels of stochastic variables (z

t ).

We model those levels through a sequential process

).

Flow-based generative models (Dinh et al., 2014; have a unique set of advantages: exact latentvariable inference, exact log-likelihood evaluation, and parallel sampling.

In flow-based generative models (Dinh et al., 2014; , we infer the latent variable z corresponding to a datapoint x, by transforming x through a composition of invertible

We assume a tractable prior p ?? (z) over latent variable z, for eg.

a Logistic or a Gaussian distribution.

By constraining the transformations to be invertible, we can compute the log-likelihood of x exactly using the change of variables rule.

Formally,

where

is transformed to h i by f i .

We learn the parameters of f 1 . . .

f K by maximizing the log-likelihood, i.e Equation (1), over a training set.

Given g = f ???1 , we can now generate a samplex from the data distribution, by sampling z ??? p ?? (z) and computingx = g(z).

We propose a generative flow for video, using the standard multi-scale flow architecture in (Dinh et al., 2016; Kingma & Dhariwal, 2018) as a building block.

In our model, we break up the latent space z into separate latent variables per timestep: z = {z t } T t=1 .

The latent variable z t at timestep t is an invertible transformation of a corresponding frame of video: x t = g ?? (z t ).

Furthermore, like in (Dinh et al., 2016; Kingma & Dhariwal, 2018) , we use a multi-scale architecture for g ?? (z t ) ( Fig. 1) : the latent variable z t is composed of a stack of multiple levels: where each level l encodes information about frame x t at a particular scale:

We first briefly describe the invertible transformations used in the multi-scale architecture to infer {z

= f ?? (x t ) and refer to (Dinh et al., 2016; Kingma & Dhariwal, 2018) for more details.

For convenience, we omit the subscript t in this subsection.

We choose invertible transformations whose Jacobian determinant in Equation 1 is simple to compute, that is a triangular matrix, diagonal matrix or a permutation matrix as explored in prior work (Rezende & Mohamed, 2015; Deco & Brauer, 1995) .

For permutation matrices, the Jacobian determinant is one and for triangular and diagonal Jacobian matrices, the determinant is simply the product of diagonal terms.

??? Actnorm:

We apply a learnable per-channel scale and shift with data-dependent initialization.

??? Coupling: We split the input y equally across channels to obtain y 1 and y 2 .

We compute z 2 = f (y 1 ) * y 2 + g(y 1 ) where f and g are deep networks.

We concat y 1 and z 2 across channels.

??? SoftPermute: We apply a 1x1 convolution that preserves the number of channels.

??? Squeeze: We reshape the input from H ?? W ?? C to H/2 ?? W/2 ?? 4C which allows the flow to operate on a larger receptive field.

We infer the latent variable z (l) at level l using:

where N is the number of steps of flow.

In Equation (3), via Split, we split the output of Flow equally across channels into h (>l) , the input to Flow (l+1) (.) and z (l) , the latent variable at level l.

We, thus enable the flows at higher levels to operate on a lower number of dimensions and larger scales.

When l = 1, h (>l???1) is just the input frame x and for l = L we omit the Split operation.

Finally, our multi-scale architecture f ?? (x t ) is a composition of the flows at multiple levels from l = 1 . . .

L from which we obtain our latent variables i.e {z

We use the multi-scale architecture described above to infer the set of corresponding latent variables for each individual frame of the video: Figure 1 for an illustration.

As in Equation (1), we need to choose a form of latent prior p ?? (z).

We use the following autoregressive factorization for the latent prior:

where z <t denotes the latent variables of frames prior to the t-th timestep: {z 1 , ..., z t???1 }.

We specify the conditional prior p ?? (z t |z <t ) as having the following factorization:

where

<t is the set of latent variables at previous timesteps and at the same level l, while z (>l) t is the set of latent variables at the same timestep and at higher levels.

See Figure 1 for a graphical illustration of the dependencies.

) be a conditionally factorized Gaussian density:

where

where N N ?? (.) is a deep 3-D residual network (He et al., 2015) augmented with dilations and gated activation units and modified to predict the mean and log-scale.

We describe the architecture and our ablations of the architecture in Section B and C of the appendix.

In summary, the log-likelhood objective of Equation (1) has two parts.

The invertible multi-scale architecture contributes

| via the sum of the log Jacobian determinants of the invertible transformations mapping the video {x t } T t=1 to {z t } T t=1 ; the latent dynamics model contributes log p ?? (z), i.e Equation (5).

We jointly learn the parameters of the multi-scale architecture and latent dynamics model by maximizing this objective.

Note that in our architecture we have chosen to let the prior p ?? (z), as described in eq. (5), model temporal dependencies in the data, while constraining the flow g ?? to act on separate frames of video.

Fooling rate SAVP-VAE 16.4 % VideoFlow 31.8 % SV2P

17.5 % Table 1 : We compare the realism of the generated trajectories using a real-vs-fake 2AFC Amazon Mechanical Turk with SAVP-VAE and SV2P.

Figure 2: We condition the VideoFlow model with the frame at t = 1 and display generated trajectories at t = 2 and t = 3 for three different shapes.

We have experimented with using 3-D convolutional flows, but found this to be computationally overly expensive compared to an autoregressive prior; in terms of both number of operations and number of parameters.

Further, due to memory limits, we found it only feasible to perform SGD with a small number of sequential frames per gradient step.

In case of 3-D convolutions, this would make the temporal dimension considerably smaller during training than during synthesis; this would change the model's input distribution between training and synthesis, which often leads to various temporal artifacts.

Using 2-D convolutions in our flow f ?? with autoregressive priors, allows us to synthesize arbitrarily long sequences without introducing such artifacts.

All our generated videos and qualitative results can be viewed at this website.

In the generated videos, a border of blue represents the conditioning frame, while a border of red represents the generated frames.

We use VideoFlow to model the Stochastic Movement Dataset used in .

The first frame of every video consists of a shape placed near the center of a 64x64x3 resolution gray background with its type, size and color randomly sampled.

The shape then randomly moves in one of eight directions with constant speed.

show that conditioned on the first frame, a deterministic model averages out all eight possible directions in pixel space.

Since the shape moves with a uniform speed, we should be able to model the position of the shape at the (t + 1) th step using only the position of the shape at the t th step.

Using this insight, we extract random temporal patches of 2 frames from each video of 3 frames.

We then use VideoFlow to maximize the loglikelihood of the second frame given the first, i.e the model looks back at just one frame.

We observe that the bits-per-pixel on the holdout set reduces to a very low 0.04 bits-per-pixel for this model.

On generating videos conditioned on the first frame, we observe that the model consistently predicts the future trajectory of the shape to be one of the eight random directions.

We compare our model with two state-of-the-art stochastic video generation models SV2P and SAVP-VAE Lee et al., 2018) using their Tensor2Tensor implementation (Vaswani et al., 2018) .

We assess the quality of the generated videos using a real vs fake Amazon Mechanical Turk test.

In the test, we inform the rater that a "real" trajectory is one in which the shape is consistent in color and congruent throughout the video.

We show that VideoFlow outperforms the baselines in terms of fooling rate in Table 1 consistently generating plausible "real" trajectories at a greater rate.

We use the action-free version of the BAIR robot pushing dataset (Ebert et al., 2017) that contain videos of a Sawyer robotic arm with resolution 64x64.

In the absence of actions, the task of video generation is completely unsupervised with multiple plausible trajectories due to the partial observability of the environment and stochasticity of the robot actions.

We train the baseline models, SAVP-VAE, SV2P and SVG-LP to generate 10 target frames, conditioned on 3 input frames.

We extract random temporal patches of 4 frames, and train VideoFlow to maximize the log-likelihood of

Bits-per-pixel VideoFlow 1.87 SAVP-VAE ??? 6.73 SV2P ??? 6.78 Table 2 : Left: We report the average bits-per-pixel across 10 target frames with 3 conditioning frames for the BAIR action-free dataset.

Figure 3 : We measure realism using a 2AFC test and diversity using mean pairwise cosine distance between generated samples in VGG perceptual space.

the 4th frame given a context of 3 past frames.

We, thus ensure that all models have seen a total of 13 frames during training.

We estimated the variational bound of the bits-per-pixel on the test set, via importance sampling, from the posteriors for the SAVP-VAE and SV2P models.

We find that VideoFlow outperforms these models on bits-per-pixel and report these values in Table 2 .

We attribute the high values of bits-per-pixel of the baselines to their optimization objective.

They do not optimize the variational bound on the log-likelihood directly due to the presence of a ?? = 1 term in their objective and scheduled sampling (Bengio et al., 2015) .

Figure 4: For a given set of conditioning frames on the BAIR action-free we sample 100 videos from each of the stochastic video generation models.

We choose the video closest to the ground-truth on the basis of PSNR, SSIM and VGG perceptual metrics and report the best possible value for each of these metrics.

All the models were trained using ten target frames but are tested to generate 27 frames.

For all the reported metrics, higher is better.

Accuracy of the best sample: The BAIR robot-pushing dataset is highly stochastic and the number of plausible futures are high.

Each generated video can be super realistic, can represent a plausible future in theory but can be far from the single ground truth video perceptually.

To partially overcome this, we follow the metrics proposed in prior work Lee et al., 2018; Denton & Fergus, 2018 ) to evaluate our model.

For a given set of conditioning frames in the BAIR action-free test-set, we generate 100 videos from each of the stochastic models.

We then compute the closest of these generated videos to the ground truth according to three different metrics, PSNR (Peak Signal to Noise Ratio), SSIM (Structural Similarity) (Wang et al., 2004) and cosine similarity using features obtained from a pretrained VGG network (Dosovitskiy & Brox, 2016; Johnson et al., 2016) and report our findings in Figure 4 .

This metric helps us understand if the true future lies in the set of all plausible futures according to the video model.

In prior work, (Lee et al., 2018; Denton & Fergus, 2018 ) effectively tune the pixel-level variance as a hyperparameter and sample from a deterministic decoder.

They obtain training stabiltiy and improve sample quality by removing pixel-level noise using this procedure.

We can remove pixel-level noise in our VideoFlow model resulting in higher quality videos at the cost of diversity by sampling videos at a lower temperature, analogous to the procedure in (Kingma & Dhariwal, 2018) .

For a network trained with additive coupling layers, we can sample the t th frame x t from P (x t |x <t ) with a temperature T simply by scaling the standard deviation of the latent gaussian distribution P (z t |z <t ) by a factor of T .

We report results with both a temperature of 1.0 and the optimal temperature tuned on the validation set using VGG similarity metrics in Figure 4 .

Additionally, we also applied low-temperature sampling to the latent gaussian priors of SV2P and SAVP-VAE and empirically found it to hurt performance.

We report these results in Figure 10 For SAVP-VAE, we notice that the hyperparameters that perform the best on these metrics are the ones that have disappearing arms.

For completeness, we report these numbers as well as the numbers for the best performing SAVP models that do not have disappearing arms.

Our model with optimal temperature performs better or as well as the SAVP-VAE and SVG-LP models on the VGG-based similarity metrics, which correlate well with human perception and SSIM.

Our model with temperature T = 1.0 is also competent with state-of-the-art video generation models on these metrics.

PSNR is explicitly a pixel-level metric, which the VAE models incorporate as part of its optimization objective.

VideoFlow on the other-hand models the conditional probability of the joint distribution of frames, hence as expected it underperforms on PSNR.

Diversity and quality in generated samples: For each set of conditioning frames in the test set, we generate 10 videos and compute the mean distance in VGG perceptual space across these 45 different pairs.

We average this across the test-set for T = 1.0 and T = 0.6 and report these numbers in Figure  3 .

We also assess the quality of the generated videos at T = 1.0 and T = 0.6, using a real vs fake Amazon Mechanical Turk test and report fooling rates.

We observe that VideoFlow outperforms diversity values reported in prior work (Lee et al., 2018) while being competitive in the realism axis.

We also find that VideoFlow at T = 0.6 has the highest fooling rate while being competent with state-of-the-art VAE models in diversity.

On inspection of the generated videos, we find that at lower temperatures, the arm exhibits less random behaviour with the background objects remaining static and clear achieving higher realism scores.

At higher temperatures, the motion of arm is much more stochastic, achieving high diversity scores with the background objects becoming much noisier leading to a drop in realism.

Figure 6: Left: We display interpolations between a) a small blue rectangle and a large yellow rectangle b) a small blue circle and a large yellow circle.

Right: We display interpolations between the first input frame and the last target frame of two test videos in the BAIR robot pushing dataset.

BAIR robot pushing dataset: We encode the first input frame and the last target frame into the latent space using our trained VideoFlow encoder and perform interpolations.

We find that the motion of the arm is interpolated in a temporally cohesive fashion between the initial and final position.

Further, we use the multi-level latent representation to interpolate representations at a particular level while keeping the representations at other levels fixed.

We find that the bottom level interpolates the motion of background objects which are at a smaller scale while the top level interpolates the arm motion.

We encode two different shapes with their type fixed but a different size and color into the latent space.

We observe that the size of the shape gets smoothly interpolated.

During training, we sample the colors of the shapes from a uniform discrete distribution which is reflected in our experiments.

We observe that all the colors in the interpolated space lie in the set of colors in the training set.

Figure 7: Left: We generate 100 frames into the future with a temperature of 0.5.

The top and bottom row correspond to generated videos in the absence and presence of occlusions respectively.

Right: We use VideoFlow to detect the plausibility of a temporally inconsistent frame to occur in the immediate future.

We generate 100 frames into the future using our model trained on 13 frames with a temperature of 0.5 and display our results in Figure 7 .

On the top, even 100 frames into the future, the generated frames remain in the image manifold maintaining temporal consistency.

In the presence of occlusions, the arm remains super-sharp but the background objects become noisier and blurrier.

Our VideoFlow model has a bijection between the z t and x t meaning that the latent state z t cannot store information other than that present in the frame x t .

This, in combination with the Markovian assumption in our latent dynamics means that the model can forget objects if they have been occluded for a few frames.

In future work, we would address this by incorporating longer memory in our VideoFlow model; for example by parameterizing N N ?? () as a recurrent neural network in our autoregressive prior (eq. 8) or using more memory-efficient backpropagation algorithms for invertible neural networks .

We use our trained VideoFlow model, conditioned on 3 frames as explained in Section 5.2, to detect the plausibility of a temporally inconsistent frame to occur in the immediate future.

We condition the model on the first three frames of a test-set video X <4 to obtain a distribution P (X 4 |X <4 ) over its 4th frame X 4 .

We then compute the likelihood of the t th frame X t of the same video to occur as the 4th time-step using this distribution.

i.e, P(X 4 = X t |X <4 ) for t = 4 . . .

13.

We average the corresponding bits-per-pixel values across the test set and report our findings in Figure 7 .

We find that our model assigns a monotonically decreasing log-likelihood to frames that are more far out in the future and hence less likely to occur in the 4th time-step.

We describe a practically applicable architecture for flow-based video prediction models, inspired by the Glow model for image generation Kingma & Dhariwal (2018) , which we call VideoFlow.

We introduce a latent dynamical system model that predicts future values of the flow model's latent state replacing the standard unconditional prior distribution.

Our empirical results show that VideoFlow achieves results that are competitive with the state-of-the-art VAE models in stochastic video prediction.

Finally, our model optimizes log-likelihood directly making it easy to evaluate while achieving faster synthesis compared to pixel-level autoregressive video models, making our model suitable for practical purposes.

In future work, we plan to incorporate memory in VideoFlow to model arbitrary long-range dependencies and apply the model to challenging downstream tasks.

be our dataset of i.i.d.

observations of a random variable x with an unknown true distribution p * (x).

Our data consist of 8-bit videos, with each dimension rescaled to the domain [0, 255/256].

We add a small amount of uniform noise to the data, u ??? U(0, 1/256.), matching its discretization level (Dinh et al., 2016; Kingma & Dhariwal, 2018) .

Let q(x) be the resulting empirical distribution corresponding to this scaling and addition of noise.

Note that additive noise is required to prevent q(x) from having infinite densities at the datapoints, which can result in ill-behaved optimization of the log-likelihood; it also allows us to recast maximization of the log-likelihood as minimization of a KL divergence.

We repeat our evaluations described in Figure 4 applying low temperature to the latent gaussian priors of SV2P and SAVP-VAE.

We empirically find that decreasing temperature from 1.0 to 0.0 monotonically decreases the performance of the VAE models.

Our insight is that the VideoFlow model gains by low-temperature sampling due to the following reason.

At lower T, we obtain a tradeoff between a performance gain by noise removal from the background and a performance hit due to reduced stochasticity of the robot arm.

On the other hand, the VAE models have a clear but slightly blurry background throughout from T = 1.0 to T = 0.0.

Reducing T in this case, solely reduces the stochasticity of the arm motion thus hurting performance.

We show correlation between training progression (measured in bits per pixel) and quality of the generated videos in Figure 11 .

We display the videos generated by conditioning on frames from the test set for three different values of bits-per-pixel on the test-set.

As we approach lower bits-per-pixel, our VideoFlow model learns to model the structure of the arm with high quality as well as its motion resulting in high quality video.

To report bits-per-pixel we use the following set of hyperparameters.

We use a learning rate schedule of linear warmup for the first 10000 steps and apply a linear-decay schedule for the last 150000 steps.

We train all our baseline models for 300K steps using the Adam optimizer.

Our models were tuned using the maximum VGG cosine similarity metric with the ground-truth across 100 decodes.

We use three values of latent loss multiplier 1e-3, 1e-4 and 1e-5.

For the SAVP-VAE model, we additionally apply linear decay on the learning rate for the last 100K steps.

SAVP-GAN: We tune the gan loss multiplier and the learning rate on a logscale from 1e-2 to 1e-4 and 1e-3 to 1e-5 respectively.

Figure 12: We compare P(X4 = Xt|X<4) and VGG cosine similarity between X4 and Xt for t = 4 . . .

13

We plot correlation between cosine similarity using a pretrained VGG network and bits-per-pixel using our trained VideoFlow model.

We compare P(X 4 = X t |X <4 ) as done in Section 5.5 and the VGG cosine similarity between X 4 and X t for t = 4 . . .

13.

We report our results for every video in the test set in Figure 13 .

We notice a weak correlation between VGG perceptual metrics and bits-per-pixel with a correlation factor of ???0.51.

I VIDEOFLOW: LOW PARAMETER REGIME We repeated our evaluations described in Figure 4 , with a smaller version of our VideoFlow model with 4x parameter reduction.

Our model remains competetive with SVG-LP on the VGG perceptual metrics.

Figure 13 : We repeat our evaluations described in Figure 4 with a smaller version of our VideoFlow model.

<|TLDR|>

@highlight

We demonstrate that flow-based generative models offer a viable and competitive approach to generative modeling of video.