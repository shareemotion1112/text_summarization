For autonomous agents to successfully operate in the real world, the ability to anticipate future scene states is a key competence.

In real-world scenarios, future states become increasingly uncertain and multi-modal, particularly on long time horizons.

Dropout based Bayesian inference provides a computationally tractable, theoretically well grounded approach to learn different hypotheses/models to deal with uncertain futures and make predictions that correspond well to observations -- are well calibrated.

However, it turns out that such approaches fall short to capture complex real-world scenes, even falling behind in accuracy when compared to the plain deterministic approaches.

This is because the used log-likelihood estimate discourages diversity.

In this work, we propose a novel Bayesian formulation for anticipating future scene states which leverages synthetic likelihoods that encourage the learning of diverse models to accurately capture the multi-modal nature of future scene states.

We show that our approach achieves accurate state-of-the-art predictions and calibrated probabilities through extensive experiments for scene anticipation on Cityscapes dataset.

Moreover, we show that our approach generalizes across diverse tasks such as digit generation and precipitation forecasting.

The ability to anticipate future scene states which involves mapping one scene state to likely future states under uncertainty is key for autonomous agents to successfully operate in the real world e.g., to anticipate the movements of pedestrians and vehicles for autonomous vehicles.

The future states of street scenes are inherently uncertain and the distribution of outcomes is often multi-modal.

This is especially true for important classes like pedestrians.

Recent works on anticipating street scenes BID13 BID9 BID23 do not systematically consider uncertainty.

Bayesian inference provides a theoretically well founded approach to capture both model and observation uncertainty but with considerable computational overhead.

A recently proposed approach BID6 BID10 uses dropout to represent the posterior distribution of models and capture model uncertainty.

This approach has enabled Bayesian inference with deep neural networks without additional computational overhead.

Moreover, it allows the use of any existing deep neural network architecture with minor changes.

However, when the underlying data distribution is multimodal and the model set under consideration do not have explicit latent state/variables (as most popular deep deep neural network architectures), the approach of BID6 ; BID10 is unable to recover the true model uncertainty (see FIG0 and BID19 ).

This is because this approach is known to conflate risk and uncertainty BID19 .

This limits the accuracy of the models over a plain deterministic (non-Bayesian) approach.

The main cause is the data log-likelihood maximization step during optimization -for every data point the average likelihood assigned by all models is maximized.

This forces every model to explain every data point well, pushing every model in the distribution to the mean.

We address this problem through an objective leveraging synthetic likelihoods BID26 BID21 which relaxes the constraint on every model to explain every data point, thus encouraging diversity in the learned models to deal with multi-modality.

In this work: 1.

We develop the first Bayesian approach to anticipate the multi-modal future of street scenes and demonstrate state-of-the-art accuracy on the diverse Cityscapes dataset without compromising on calibrated probabilities, 2.

We propose a novel optimization scheme for dropout based Bayesian inference using synthetic likelihoods to encourage diversity and accurately capture model uncertainty, 3.

Finally, we show that our approach is not limited to street scenes and generalizes across diverse tasks such as digit generation and precipitation forecasting.

Bayesian deep learning.

Most popular deep learning models do not model uncertainty, only a mean model is learned.

Bayesian methods BID15 BID18 on the other hand learn the posterior distribution of likely models.

However, inference of the model posterior is computationally expensive.

In BID6 this problem is tackled using variational inference with an approximate Bernoulli distribution on the weights and the equivalence to dropout training is shown.

This method is further extended to convolutional neural networks in BID5 .

In BID10 this method is extended to tackle both model and observation uncertainty through heteroscedastic regression.

The proposed method achieves state of the art results on segmentation estimation and depth regression tasks.

This framework is used in BID2 to estimate future pedestrian trajectories.

In contrast, BID22 propose a (unconditional) Bayesian GAN framework for image generation using Hamiltonian Monte-Carlo based optimization with limited success.

Moreover, conditional variants of GANs BID17 are known to be especially prone to mode collapse.

Therefore, we choose a dropout based Bayesian scheme and improve upon it through the use of synthetic likelihoods to tackle the issues with model uncertainty mentioned in the introduction.

Structured output prediction.

Stochastic feedforward neural networks (SFNN) and conditional variational autoencoders (CVAE) have also shown success in modeling multimodal conditional distributions.

SFNNs are difficult to optimize on large datasets BID25 due to the binary stochastic variables.

Although there has been significant effort in improving training efficiency BID20 BID7 , success has been partial.

In contrast, CVAEs BID24 assume Gaussian stochastic variables, which are easier to optimize on large datasets using the re-parameterization trick.

CVAEs have been successfully applied on a large variety of tasks, include conditional image generation BID1 , next frame synthesis BID28 , video generation BID0 BID4 , trajectory prediction BID12 among others.

The basic CVAE framework is improved upon in BID3 through the use of a multiple-sample objective.

However, in comparison to Bayesian methods, careful architecture selection is required and experimental evidence of uncertainty calibration is missing.

Calibrated uncertainties are important for autonomous/assisted driving, as users need to be able to express trust in the predictions for effective decision making.

Therefore, we also adopt a Bayesian approach over SFNN or CVAE approaches.

Anticipation future scene scenes.

In BID13 ) the first method for predicting future scene segmentations has been proposed.

Their model is fully convolutional with prediction at multiple scales and is trained auto-regressively.

BID9 improves upon this through the joint prediction of future scene segmentation and optical flow.

Similar to BID13 a fully convolutional model is proposed, but the proposed model is based on the Resnet-101 BID8 and has a single prediction scale.

More recently, BID14 has extended the model of BID13 to the related task of future instance segmentation prediction.

These methods achieve promising results and establish the competence of fully convolutional models.

In BID23 a Convolutional LSTM based model is proposed, further improving short-term results over BID9 .

However, fully convolutional architectures have performed well at a variety of related tasks, including segmentation estimation BID29 BID30 , RGB frame prediction BID16 BID0 among others.

Therefore, we adopt a standard ResNet based fully-convolutional architecture, while providing a full Bayesian treatment.

We phrase our models in a Bayesian framework, to jointly capture model (epistemic) and observation (aleatoric) uncertainty BID10 .

We begin with model uncertainty.

Let x ∈ X be the input (past) and y ∈ Y be the corresponding outcomes.

Consider f : x → y, we capture model uncertainty by learning the distribution p(f |X, Y) of generative models f , likely to have generated our data {X, Y}. The complete predictive distribution of outcomes y is obtained by marginalizing over the posterior distribution, DISPLAYFORM0 However, the integral in FORMULA0 is intractable.

But, we can approximate it in two steps BID6 .

First, we assume that our models can be described by a finite set of variables ω.

Thus, we constrain the set of possible models to ones that can be described with ω.

Now, (1) is equivalently, DISPLAYFORM1 Second, we assume an approximating variational distribution q(ω) of models which allows for efficient sampling.

This results in the approximate distribution, DISPLAYFORM2 For convolutional models, BID5 proposed a Bernoulli variational distribution defined over each convolutional patch.

The number of possible models is exponential in the number of patches.

This number could be very large, making it difficult optimize over this very large set of models.

In contrast, in our approach (4), the number possible models is exponential in the number of weight parameters, a much smaller number.

In detail, we choose the set of convolutional kernels and the biases {( DISPLAYFORM3 of our model as the set of variables ω.

Then, we define the following novel approximating Bernoulli variational distribution q(ω) independently over each element w i,j k ,k (correspondingly b k ) of the kernels and the biases at spatial locations {i, j}, DISPLAYFORM4 Note, denotes the hadamard product, M k are tuneable variational parameters, z i,j k ,k ∈ Z K are the independent Bernoulli variables, p K is a probability tensor equal to the size of the (bias) layer, |K| (|K |) is the number of kernels in the current (previous) layer.

Here, p K is chosen manually.

Moreover, in contrast to BID5 , the same (sampled) kernel is applied at each spatial location leading to the detection of the same features at varying spatial locations.

Next, we describe how we capture observation uncertainty.

Observation uncertainty can be captured by assuming an appropriate distribution of observation noise and predicting the sufficient statistics of the distribution BID10 .

Here, we assume a Gaussian distribution with diagonal covariance matrix at each pixel and predict the mean vector µ i,j and co-variance matrix σ i,j of the distribution.

In detail, the predictive distribution of a generative model draw fromω ∼ q(ω) at a pixel position {i, j} is, DISPLAYFORM0 We can sample from the predictive distribution p(y|x) (3) by first sampling the weight matrices ω from (4) and then sampling from the Gaussian distribution in (5).

We perform the last step by the linear transformation of a zero mean unit diagonal variance Gaussian, ensuring differentiability, DISPLAYFORM1 where,ŷ i,j is the sample drawn at a pixel position {i, j} through the liner transformation of z (a vector) with the predicted mean µ i,j and variance σ i,j .

In case of street scenes, y i,j is a class-confidence vector and sample of final class probabilities is obtained by pushingŷ i,j through a softmax.

For a good variational approximation (3), our approximating variational distribution of generative models q(ω) should be close to the true posterior p(ω|X, Y).

Therefore, we minimize the KL divergence between these two distributions.

As shown in BID6 a) ; BID10 the KL divergence is given by (over i.i.d data points), DISPLAYFORM0 The log-likelihood term at the right of FORMULA7 considers every model for every data point.

This imposes the constraint that every data point must be explained well by every model.

However, if the data distribution (x, y) is multi-modal, this would push every model to the mean of the multi-modal distribution (as in FIG0 where only way for models to explain both modes is to converge to the mean).

This discourages diversity in the learned modes.

In case of multi-modal data, we would not be able to recover all likely models, thus hindering our ability to fully capture model uncertainty.

The models would be forced to explain the data variation as observation noise BID19 , thus conflating model and observation uncertainty.

We propose to mitigate this problem through the use of an approximate objective using synthetic likelihoods BID26 BID21 ) -obtained from a classifier.

The classifier estimates the likelihood based on whether the modelsω ∼ q(ω) explain (generate) data samples likely under the true data distribution p(y|x).

This removes the constraint on models to explain every data point -it only requires the explained (generated) data points to be likely under the data distribution.

Thus, this allows modelsω ∼ q(ω) to be diverse and deal with multi-modality.

Next, we reformulate the KL divergence estimate of (7) to a likelihood ratio form which allows us to use a classifier to estimate (synthetic) likelihoods, (also see Appendix), DISPLAYFORM1 In the second step of FORMULA8 , we divide and multiply the probability assigned to a data sample by a model p(y|x, ω) by the true conditional probability p(y|x) to obtain a likelihood ratio.

We can estimate the KL divergence by equivalently estimating this ratio rather than the true likelihood.

In order to (synthetically) estimate this likelihood ratio, let us introduce the variable θ to denote, p(y|x, θ = 1) the probability assigned by our model ω to a data sample (x, y) and p(y|x, θ = 0) the true probability of the sample.

Therefore, the ratio in the last term of FORMULA8 is, DISPLAYFORM2 In the last step of FORMULA9 we use the fact that the events θ = 1 and θ = 0 are mutually exclusive.

We can approximate the ratio p(θ=1|x,y)1−p(θ=1|x,y) by jointly learning a discriminator D(x,ŷ) that can distinguish between samples of the true data distribution and samples (x,ŷ) generated by the model ω, which provides a synthetic estimate of the likelihood, and equivalently integrating directly over (x,ŷ), DISPLAYFORM3 Note that the synthetic likelihood DISPLAYFORM4 is independent of any specific pair (x, y) of the true data distribution (unlike the log-likelihood term in FORMULA7 ), its value depends only upon whether the generated data point (x,ŷ) by the model ω is likely under the true data distribution p(y|x).

Therefore, the models ω have to only generate samples (x,ŷ) likely under the true data distribution.

The models need not explain every data point equally well.

Therefore, we do not push the models ω to the mean, thus allowing them to be diverse and allowing us to better capture uncertainty.

Empirically, we observe that a hybrid log-likelihood term using both the log-likelihood terms of FORMULA0 and FORMULA7 with regularization parameters α and β (with α ≥ β) stabilizes the training process, DISPLAYFORM5 Note that, although we do not explicitly require the posterior model distribution to explain all data points, due to the exponential number of models afforded by dropout and the joint optimization (min-max game) of the discriminator, empirically we see very diverse models explaining most data points.

Moreover, empirically we also see that predicted probabilities remain calibrated.

Next, we describe the architecture details of our generative models ω and the discriminator D(x,ŷ).

The architecture of our ResNet based generative models in our model distribution q(ω) is shown in FIG1 .

The generative model takes as input a sequence of past segmentation class-confidences s p , the past and future vehicle odometry o p , o f (x = {s p , o p , o f }) and produces the class-confidences at the next time-step as output.

The additional conditioning on vehicle odometry is because the sequences are recorded in frame of reference of a moving vehicle and therefore the future observed sequence is dependent upon the vehicle trajectory.

We use recursion to efficiently predict a sequence of future scene segmentations y = {s f }.

The discriminator takes as input s f and classifies whether it was produced by our model or is from the true data distribution.

In detail, generative model architecture consists of a fully convolutional encoder-decoder pair.

This architecture builds upon prior work of BID13 BID9 , however with key differences.

In BID13 , each of the two levels of the model architecture consists of only five convolutional layers.

In contrast, our model consists of one level with five convolutaional blocks.

The encoder contains three residual blocks with max-pooling in between and the decoder consists of a residual and a convoluational block with up-sampling in between.

We double the size of the blocks following max-pooling in order to preserve resolution.

This leads to a much deeper model with fifteen convolutional layers, with constant spatial convolutional kernel sizes.

This deep model with pooling creates a wide receptive field and helps better capture spatio-temporal dependencies.

The residual connections help in the optimization of such a deep model.

Computational resources allowing, it is possible to add more levels to our model.

In BID9 a model is considered which uses a Res101-FCN as an encoder.

Although this model has significantly more layers, it also introduces a large amount of pooling.

This leads to loss of resolution and spatial information, hence degrading performance.

Our discriminator model consists of six convolutional layers with max-pooling layers in-between, followed by two fully connected layers.

Finally, in Appendix E we provide layer-wise details and discuss the reduction of number of models in q(ω) through the use of Weight Dropout (4) for our architecture of generators.

Next, we evaluate our approach on MNIST digit generation and street scene anticipation on Cityscapes.

We further evaluate our model on 2D data FIG0 ) and precipitation forecasting in the Appendix.

Here, we aim to generate the full MNIST digit given only the lower left quarter of the digit.

This task serves as an ideal starting point as in many cases there are multiple likely completions given the lower left quarter digit, e.g. 5 and 3.

Therefore, the learned model distribution q(ω) should contain likely models corresponding to these completions.

We use a fully connected generator with 6000-4000-2000 hidden units with 50% dropout probability.

The discriminator has 1000-1000 hidden units with leaky ReLU non-linearities.

We set β = 10 −4 for the first 4 epochs and then reduce it to 0, to provide stability during the initial epochs.

We compare our synthetic likelihood based approach (Bayes-SL) with, 1.

A non-Bayesian mean model, 2.

A standard Bayesian approach (Bayes-S), 3.

A Conditional Variational Autoencoder (CVAE) (architecture as in BID24 ).

As evaluation metric we consider (oracle) Top-k% accuracy BID12 .

We use a standard Alex-Net based classifier to measure if the best prediction corresponds to the ground-truth class -identifies the correct mode -in TAB1 (right) over 10 splits of the MNIST test-set.

We sample 10 models from our learned distribution and consider the best model.

We see that our Bayes-SL performs best, even outperforming the CVAE model.

In the qualitative examples in TAB1 (left), we see that generations from modelsω ∼ q(ω) sampled from our learned model distribution corresponds to clearly defined digits (also in comparision to FIG2 in BID24 ).

In contrast, we see that the Bayes-S model produces blurry digits.

All sampled models have been pushed to the mean and shows little advantage over a mean model.

Next, we evaluate our apporach on the Cityscapes dataset -anticipating scenes more than 0.5 seconds into the future.

The street scenes already display considerable multi-modality at this time-horizon.

Evaluation metrics and baselines.

We use PSPNet BID30 to segment the full training sequences as only the 20 th frame has groundtruth annotations.

We always use the annotated 20 th frame of the validation sequences for evaluation using the standard mean Intersection-over-Union (mIoU) and the per-pixel (negative) conditional log-likelihood (CLL) metrics.

We consider the following baselines for comparison to our Resnet based (architecture in FIG1 ) Bayesian (Bayes-WD-SL) model with weight dropout and trained using synthetic likelihoods: 1.

Copying the last seen input; 2.

A non-Bayesian (ResG-Mean) version; 3.

A Bayesian version with standard patch dropout (Bayes-S); 4.

A Bayesian version with our weight dropout (Bayes-WD).

Note that, combination of ResG-Mean with an adversarial loss did not lead to improved results (similar observations made in BID13 ).

We use grid search to set the dropout rate (in (4)) to 0.15 for the Bayes-S and 0.20 for Bayes-WD(-SL) models.

We set α, β = 1 for our Bayes-WD-SL model.

We train all models using Adam BID11 for 50 epochs with batch size 8.

We use one sample to train the Bayesian methods as in BID5 and use 100 samples during evaluation.

Comparison to state of the art.

We begin by comparing our Bayesian models to state-of-the-art methods BID13 ; BID23 in TAB0 .

We use the mIoU metric and for a fair comparison consider the mean (of all samples) prediction of our Bayesian models.

We alwyas compare to the groundtruth segmentations of the validation set.

However, as all three methods use a slightly different semantic segmentation algorithm (Table 2) to generate training and input test data, we include the mIoU achieved by the Last Input of all three methods (see Appendix C for results using Dialation 10).

Similar to Luc et al. FORMULA0 we fine-tune (ft) to predict at 3 frame intervals for better performance at +0.54sec.

Our Bayes-WD-SL model outperforms baselines and improves on prior work by 2.8 mIoU at +0.06sec and 4.8 mIoU/3.4 mIoU at +0.18sec/+0.54sec respectively.

Our Bayes-WD-SL model also obtains higher relative gains in comparison to BID13 with respect the Last Input Baseline.

These results validate our choice of model architecture and show that our novel approach clearly outperforms the state-of-the-art.

The performance advantage of Bayes-WD-SL over Bayes-S shows that the ability to better model uncertainty does not come at the cost of lower mean performance.

However, at larger time-steps as the future becomes increasingly uncertain, mean predictions (mean of all likely futures) drift further from the ground-truth.

Therefore, next we evaluate the models on their (more important) ability to capture the uncertainty of the future.

Evaluation of predicted uncertainty.

Next, we evaluate whether our Bayesian models are able to accurately capture uncertainity and deal with multi-modal futures, upto t + 10 frames (0.6 seconds) in TAB1 .

We consider the mean of (oracle) best 5% of predictions BID12 ) of our Bayesian models to evaluate whether the learned model distribution q(ω) contains likely models corresponding to the groundtruth.

We see that the best predictions considerably improve over the mean predictionsshowing that our Bayesian models learns to capture uncertainity and deal with multi-modal futures.

Quantitatively, we see that the Bayes-S model performs worst, demonstrating again that standard dropout BID10 struggles to recover the true model uncertainity.

The use of weight dropout improves the performance to the level of the ResG-Mean model.

Finally, we see that our Bayes-WD-SL model performs best.

In fact, it is the only Bayesian model whose (best) performance exceeds that of the ResG-Mean model (also outperforming state-of-the-art), demonstrating the effectiveness of synthetic likelihoods during training.

In FIG3 we show examples comparing the best prediction of our Bayes-WD-SL model and ResG-Mean at t + 9.

The last row highlights the differences between the predictions -cyan shows areas where our Bayes-WD-SL is correct and ResG-Mean is wrong, red shows the opposite.

We see that our Bayes-WD-SL performs better at classes like cars and pedestrians which are harder to predict (also in comparison to TAB3 in BID13 ).

In FIG4 , we show samples from randomly sampled modelsω ∼ q(ω), which shows correspondence to the range of possible movements of bicyclists/pedestrians.

Next, we further evaluate the models with the CLL metric in TAB1 .

We consider the mean predictive distributions (3) up to t + 10 frames.

We see that the Bayesian models outperform the ResG-Mean model significantly.

In particular, we see that our Bayes-WD-SL model performs the best, demonstrating that the learned model and observation uncertainty corresponds to the variation in the data.

Comparison to a CVAE baseline.

As there exists no CVAE BID24 based model for future segmentation prediction, we construct a baseline as close as possible to our Bayesian models Groundtruth, t + 9 ResG-Mean, t + 9 Bayes-WD-SL, t + 9 Comparison Sample #1, t + 9 Sample #2, t + 9 Sample #3, t + 9 Sample #4, t + 9 based on existing CVAE based models for related tasks BID0 BID28 .Existing CVAE based models BID0 BID28 ) contain a few layers with Gaussian input noise.

Therefore, for a fair comparison we first conduct a study in TAB2 to find the layers which are most effective at capturing data variation.

We consider Gaussian input noise applied in the first, middle or last convolutional blocks.

The noise is input dependent during training, sampled from a recognition network (see Appendix).

We observe that noise in the last layers can better capture data variation.

This is because the last layers capture semantically higher level scene features.

Overall, our Bayesian approach (Bayes-WD-SL) performs the best.

This shows that the CVAE model is not able to effectively leverage Gaussian noise to match the data variation.

Uncertainty calibration.

We further evaluate predicted uncertainties by measuring their calibration -the correspondence between the predicted probability of a class and the frequency of its occurrence in the data.

As in BID10 , we discretize the output probabilities of the mean predicted distribution into bins and measure the frequency of correct predictions for each bin.

We report the results at t + 10 frames in FIG5 .

We observe that all Bayesian approaches outperform the ResG-Mean and CVAE versions.

This again demonstrates the effectiveness of the Bayesian approaches in capturing uncertainty.

We propose a novel approach for predicting real-world semantic segmentations into the future that casts a convolutional deep learning approach into a Bayesian formulation.

One of the key contributions is a novel optimization scheme that uses synthetic likelihoods to encourage diversity and deal with multi-modal futures.

Our proposed method shows state of the art performance in challenging street scenes.

More importantly, we show that the probabilistic output of our deep learning architecture captures uncertainty and multi-modality inherent to this task.

Furthermore, we show that the developed methodology goes beyond just street scene anticipation and creates new opportunities to enhance high performance deep learning architectures with principled formulations of Bayesian inference.

KL divergence estimate.

Here, we provide a detailed derivation of (8).

Starting from (7), we have, DISPLAYFORM0 Multiplying and dividing by p(y|x), the true probability of occurance, DISPLAYFORM1 Using q(ω) dω = 1, DISPLAYFORM2 As log p(y|x)d(x, y) is independent of ω, the variables we are optmizing over, we have, DISPLAYFORM3 APPENDIX B. RESULTS ON SIMPLE MULTI-MODAL 2D DATA.

We show results on simple multi-modal 2d data as in the motivating example in the introduction.

The data consists of two parts: x ∈ [−10, 0] we have y = 0 and x ∈ [0, 10] we have y = (−0.3, 0.3).The set of models under consideration is a two hidden layer neural network with 256-128 neurons with 50% dropout.

We show 10 randomly sampled models fromω ∼ q(ω) learned by the Bayes-S approach in FIG6 and our Bayes-SL approach in FIG7 (with α = 1, β = 0).

We assume constant observation uncertainty (=1).

We clearly see that our Bayes-SL learns models which cover both modes, while all the models learned by Bayes-S fit to the mean.

Clearly showing that our approach can better capture model uncertainty.

First, we provide additional training details of our Bayes-WD-SL in TAB3 .

Generator learning rate 1 × 10 DISPLAYFORM0 Discriminator learning rate 1 × 10 DISPLAYFORM1 # Generator updates per iteration 1 # Discriminator updates per iteration 1 Table 6 : Additional Comparison to BID13 using the same Dialation 10 approach to generate training segmentations.

Note: Fine Tuned (ft) means both approaches are trained to predict at intervals of three frames (0.18 seconds).Second, we provide additional evaluation on street scenes.

In Section 4.2 TAB0 we use a PSPNet to generate training segmentations for our Bayes-WD-SL model to ensure fair comparison with the state-of-the-art BID23 .

However, the method of BID13 uses a weaker Dialation 10 approach to generate training segmentations.

Note that our Bayes-WD-SL model already obtains higher gains in comparison to BID13 with respect the Last Input Baseline, e.g. at +0.54sec, 47.8 -36.9 = 10.9 mIoU translating to 29.5% gain over the Last Input Baseline of BID13 versus 51.2 -38.3 = 12.9 mIoU translating to 33.6% gain over the Last Input Baseline of our Bayes-WD-SL model in TAB0 .

But for fairness, here we additionally include results in Table 6 using the same Dialation 10 approach to generate training segmentations.

We observe that our Bayes-WD-SL model beats the model of BID13 in both short-term (+0.18 sec) and long-term predictions (+0.54 sec).

Furthermore, we see that the mean of the Top 5% of the predictions of Bayes-WD-SL leads to much improved results over mean predictions.

This again confirms the ability of our Bayes-WD-SL model to capture uncertainty and deal with multi-modal futures.

APPENDIX D. RESULTS ON HKO PRECIPITATION FORECASTING DATA.The HKO radar echo dataset consists of weather radar intensity images.

We use the train/test split used in Xingjian et al. FORMULA0 ; BID3 .

Each sequence consists of 20 frames.

We use 5 frames as input and 15 for prediction.

Each frame is recorded at an interval of 6 minutes.

Therefore, they display considerable uncertainty.

We use the same network architecture as used for street scene segmentation Bayes-WD-SL FIG1 and with α = 5, β = 1), but with half the convolutional filters at each level.

We compare to the following baselines: 1.

A deterministic model (ResG-Mean), 2.

A Bayesian model with weight dropout.

We report the (oracle) Top-10% scores (best 1 of 10), over the following metrics BID27 BID3 ), 1.

Rainfall-MSE: Rainfall mean squared error, 2.

CSI: Critical success index, 3.

FAR: False alarm rate, 4.

POD: Probability of detection, and 5.

Correlation, in Table 7 , Note, that Xingjian et al. (2015) ; BID3 reports only scores over mean of all samples.

Our ResG-Mean model outperforms these state of the art methods, showing the versatility of our model architecture.

Our Bayes-WD-SL can outperform the strong ResG-Mean baseline again showing that it learns to capture uncertainty (see FIG0 ).

In comparison, the Bayes-WD baseline struggles to outperform the ResG-Mean baseline.

We further compare the calibration our Bayes-SL model to the ResG-Mean model in FIG8 .

We plot the predicted intensity to the true mean observed intensity.

Table 7 : Evaluation on HKO radar image sequences.

is stark in the high intensity region.

The RegG-Mean model deviates strongly from the diagonal in this region -it overestimates the radar intensity.

In comparison, we see that our Bayes-WD-SL approach stays closer to the diagonal.

These results again show that our synthetic likelihood based approach leads to more accurate predictions while not compromising on calibration.

Observation Groundtruth Observation Prediction t − 5 t − 3 t − 1 t t + 2 t + 4 t + 6 t + 8 t + 10 t + 12 t + 13 t + 14

Here, we provide layer-wise details of our generative and discriminative models in TAB6 .

We provide layer-wise details of the recognition network of the CVAE baseline used in TAB2 (in the main paper).

Finally, in TAB0 we show the difference in the number of possible models using our weight based variational distribution 4 (weight dropout) versus the patch based variational distribution (patch dropout) proposed in BID5 .

The number of patches is calculated using the formula, DISPLAYFORM0 because we use convolutional stride 1, padding to ensure same output resolution and each patch is dropped out (in BID5 ) independently for each convolutional filter.

The number of weight parameters is given by the formula,Filter size × # Input Convolutional Filters × # Output Convolutional Filters + # Bias.

TAB0 shows that our weight dropout scheme results in significantly lower number of parameters compared to patch dropout BID5 .Details of our generative model.

We show the layer wise details in TAB6 Details of our discriminator model.

We show the layer wise details in TAB8 .Details of the recognition model used in the CVAE baseline.

We show the layer wise details in TAB0 .

36,700,406 11,699,192 36,700,406 11,699,192 Table 11: The difference in the number of possible models using our Weight dropout scheme versus patch dropout BID5 (Appendix D) 18,350,203 5,849,596 68.1% TAB0 : Overview of the variational parameters using our Weight dropout scheme versus patch dropout BID5 of both architectures for street scene and precipitation forecasting

<|TLDR|>

@highlight

Dropout based Bayesian inference is extended to deal with multi-modality and is evaluated on scene anticipation tasks.