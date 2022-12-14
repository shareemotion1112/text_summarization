In many robotic applications, it is crucial to maintain a belief about the state of  a system, like the location of a robot or the pose of an object.

These state estimates serve as input for planning and decision making and  provide feedback during task execution.

Recursive Bayesian Filtering algorithms address the state estimation problem, but they require a model of the process dynamics and the sensory observations as well as  noise estimates that quantify the accuracy of these models.

Recently, multiple works have demonstrated that the process and sensor models can be  learned by end-to-end training through differentiable versions of Recursive Filtering methods.

However, even if the predictive models are known, finding suitable noise models  remains challenging.

Therefore, many practical applications rely on very simplistic noise  models.

Our hypothesis is that end-to-end training through differentiable Bayesian  Filters enables us to learn more complex heteroscedastic noise models for the system dynamics.

We evaluate learning such models with different types of  filtering algorithms and on two different robotic tasks.

Our experiments show that especially  for sampling-based filters like the Particle Filter, learning heteroscedastic noise  models can drastically improve the tracking performance in comparison to using  constant noise models.

For many real-world systems that we would like to control, we cannot directly observe the current state directly.

However, in order to stabilize a system at a goal state or make it track a trajectory, we need to have access to state feedback.

An observer provides an estimate of the current system state from sensor measurements.

Recursive Bayesian Filtering is a probabilistic approach towards estimating a belief about the current state.

The method relies on a process model that predicts how the system behaves over time and an observation model that generates the expected observations given the predicted state.

While the approach itself is general and makes few assumptions, the challenge is to formulate the process and observation models and to estimate the noise in these models.

Process and observation noise quantify how certain the filter is about either the prediction or the observations.

This information is used to determine how much the predicted state is updated based on the observation.

Deep neural networks are well suited for tasks that require finding patterns or extracting information from raw, high-dimensional input signals and compressing them into a more compact representation.

They have therefore become the method of choice especially in perception problems.

For many robotics tasks like modeling dynamics, planning or tracking however, it has been shown that combining prior knowledge in the form of analytical models and/or algorithmic structure with trainable network components leads to better performance and generalizability than trying to learn the complete tasks from scratch BID17 BID11 BID9 BID23 BID19 BID8 BID6 BID12 .Specifically, BID8 BID6 BID9 BID12 have presented differentiable Bayesian Filtering algorithms.

The authors focus on learning the observation and dynamics models end-to-end through the filters and demonstrate that the recursive filtering structure improves prediction results over using recurrent neural networks that were trained for the same task.

In many robotic applications, it is possible to formulate the process and observation model based on first-order principles.

However, finding appropriate values for the process and observation noise is often difficult and despite of much research on identification methods (e.g. BID2 BID25 ) they are often tuned manually.

To reduce the tedious tuning effort, the noise models are typically assumed to be a Gaussian with zero mean and constant covariance.

Many real systems can however be better modeled with heteroscedastic noise models, where the level of uncertainty depends on the state of the system and/or possible control inputs.

Taking heterostochasticity into account has been demonstrated to improve filtering performance in many robotic tasks BID1 BID14 .In this work, we propose a method to learn heteroscedastic noise models from data by optimizing the prediction likelihood end-to-end through differentiable Bayesian Filters.

In addition to differentiable Extended Kalman Filters and Particle Filters, which have been proposed in related work, we also propose two different versions of the Unscented Kalman Filter.

In our experiments we focus on learning the noise models and therefore assume that observation and process models are known or at least pretrained.

We evaluate the performance of the different filters and noise models on two different real-world robotic problems: (i) Visual Odometry for an driving car BID6 BID9 BID4 which has simple smooth dynamics and a low-dimensional state, and (ii) Visual tracking of an object that is pushed by a robot (Yu et al., 2016; BID17 .

Planar pushing has challenging, discontinuous dynamics and was shown to have a heteroscedastic noise distribution BID1 .

Furthermore, the dimensionality of the state is double of the Visual Odometry task.

Our experiments show that using heteroscedastic process noise models drastically improves the tracking performance of the Particle Filter and Unscented Filter variants and facilitated learning as compared to learning a constant process noise model.

While learning the noise models can be beneficial for all filters, the tracking performance of the EKF turned out to be least sensitive to the noise models.

In comparison to the process noise, learning the observation noise did not improve the results much for the two tasks we evaluated.

Filtering refers to the problem of estimating the state x of a stochastic system at time step t given an initial believe x 0 , a sequence of observations z t and control inputs u t .

The aim is to compute p(x t |x 0...t???1 , u 0...t , z 0...t ).

To do so, we describe the system with a state space representation, that consists of two equations: The process model f describes how the state changes over time and the observation model h generates observations given the current state: DISPLAYFORM0 The random variables q and r are the process and observation noise and represent the stochasticity of the system.

This model makes the Markov assumption, i.e. the current state only depends on the previous state, and the observation only depends on the current state.

This assumption makes it possible to compute p(x t |x 0.

DISPLAYFORM1 In the following, we review the most common filtering algorithms.

For more details, we refer to BID24 .

The Kalman Filter BID10 ) is a closed-form solution to the filtering problem for systems with linear process and observation model and Gaussian additive noise.

DISPLAYFORM0 Given these assumptions and a Gaussian initial belief, the belief can be represented by the mean ?? and covariance matrix ?? over the estimate.

At each timestep, the filter predicts?? and?? given the process model.

The innovation i t is the difference between the predicted and actual observation and is used to correct the prediction.

The Kalman Gain K trades-off the process noise Q and the observation noise R to determine the magnitude of the update.

Step: UpdateStep: DISPLAYFORM0

The EKF BID22 extends the Kalman Filter to systems with non-linear process and observation models.

It uses the non-linear models for predicting?? and the corresponding observations??? (Equations 1, 5).

For computing the prediction and update of ?? and K, these models are linearized around the current mean of the state and the Jacobians F |??t and H |??t replace A and H in Equations 2 -4 and 7.

This first-order approximation can be problematic for systems with strong non-linearity, as it does not take the uncertainty about the mean into account BID26 ).

The UKF (Simon J. BID21 BID26 ) was proposed to address the aforementioned problem of the EKF.

Its core idea is to represent a Gaussian random variable by a set of specifically chosen points in state space, the so called sigma points X .

If this random variable undergoes a nonlinear transformation, we can calculate its new statistics from the transformed sigma points.

This method is called the Unscented Transform (Simon J. Julier, 1997).

For example, in the prediction step of the UKF, the non-linear transform is the process model (Equation 10) and the new mean and covariance are computed in Equations 11 and 12 DISPLAYFORM0 DISPLAYFORM1 By applying the non-linear prediction step separately to each sigma point and then fitting a new Gaussian to the transformed points (Equations 11, 12) , the UKF conveys the non-linear transformation of the covariance more faithfully than the EKF and is thus better suited for strongly non-linear problems BID24 .The parameter ?? controls the spread of the sigma points and how strongly the original mean X 0 is weighted in comparison to the other sigma points.

In practice, we found ?? difficult to tune since placing the sigma points too far from the mean increases prediction uncertainty and can even destabilize the filter.

Simon J. Julier (1997) suggested to chose ?? such that ?? + n = 3.

This however results in negative values of ?? if n > 3, for which the estimated covariance matrix is not guaranteed to be positive semidefinite anymore BID21 .

In addition, X 0 , which represents the original mean, is weighted negatively in this case, which seems counterintuitive and can cause divergence of the estimated mean.

The UKF represents the belief over the state with as few sigma points as possible.

However, as described above, finding the correct scaling parameter ?? can be difficult, especially if the state is high dimensional.

Instead of relying on the unscented transform to calculate the mean and covariance of the state at the next timestep, we can also resort to Monte Carlo methods, as proposed by W??thrich et al. (2016) .

In practice, this means that we replace the carefully constructed sigma points and their weights in Equations 8, 9 with samples from the current estimated state distribution, which all have uniform weights.

The rest of the UKF algorithm stays the same, but more samples are necessary to represent the distribution accurately.

In contrast to the different variants of the Kalman Filter explained before, the Particle Filter BID5 does not assume a parametric representation of the state distribution.

Instead, it represents the state with a set of particles.

The particle-based representation allows the filter to track multiple hypotheses about the state at the same time and makes it a popular choice for tasks like localization or visual object tracking BID24 ).An initial set of particles X 0 is drawn from some prior belief and initialized with uniform weights.

At each recursion step, new particles are generated by sampling process noise and applying the process model to the previous particles: DISPLAYFORM0 For each observation z t , we then evaluate the likelihood p(z t |x i t ) of a particle x i t having generated this observation.

Based on this, the weight w i of each particle is updated: DISPLAYFORM1 .

A common problem of this filter is particle deprivation: Over time, many particles will receive a very low likelihood p(z t |x i t ), and the state would be represented by too few particles with high weights.

To prevent this, the particle filter algorithm uses resampling, where a new set of particles with uniform weights is drawn (with replacement) from the old set, according to the weights.

This step focuses the particle set on regions of high likelihood and is usually applied after each timestep.3 RELATED WORK DIFFERENTIABLE FILTERING Haarnoja et al. (2016) proposed the BackpropKF, a differentiable implementation of the Kalman Filter.

While the observation and process model were assumed to be known, the differentiable implementation enabled the authors to train a neural network through the filter to preprocess the input images.

This network can be viewed as a trainable part of the sensor which extracts the relevant information from the high-dimensional raw input data and also predicts the observation noise R dependent on the images.

This heteroscedastic observation noise model was shown to be useful in situations where the desired information could not be extracted from the image, e.g. when a tracked object is occluded.

BackpropKF outperformed an LSTM model that was trained to perform the same tasks due to the prior knowledge encoded in the Filtering algorithm and the given models.

Jonschkowski & Brock (2016) presented a differentiable Histogram Filter for discrete localization tasks in one or two dimensions.

For this low-dimensional problem , both, the observation and the process model, were trained through the filter in a supervised or unsupervised manner.

Experiments showed that optimizing the models end-to-end through the filter improved results on the metric that was optimized during training (MSE or localization accuracy) in comparison to filtering with models that were trained in isolation.

BID9 ; BID12 proposed differentiable Particle Filters for localization and tracking of a mobile robot.

In each work, a neural network was trained to predict the likelihood p(z t |x i t ) of each particle given an image and a map of the environment.

While BID12 used a given process model, BID9 learned the process model and the distribution from which the process noise is sampled.

They however did not evaluate their method when only the process model or only the noise was learned and it is thus not clear how each of these two components individually affected the overall error rate of the filter.

BID12 additionally introduced soft resampling and thereby enabled backpropagation through more than one time step.

Related work demonstrated that (i) integrating algorithmic structure with learning leads to better results than training unconstrained networks and that (ii) it is possible and beneficial to train the components of the filters end-to-end instead of in isolation.

Each work focused on creating a differentiable version of a particular filtering algorithm.

In this work, we propose to learn heteroscedastic noise models and analyze the benefit of these models within different filtering algorithms and for two different applications.

All previous work described above was evaluated on tracking and visual odometry problems with low-dimensional states and observations (at most five dimensions) and smooth, although non-linear dynamics models.

In contrast to this, we additionally evaluate the methods on a planar pushing task which has challenging non-linear and discontinuous dynamics due to physical contact and a 10-dimensional state space.

Variational methods provide an alternative way of learning the parameters of a probabilistic generative model and performing inference on its latent states.

The main idea of variational inference is to approximate the intractable posterior distribution p ?? (x|z) by an approximate distribution q ?? (x|z) that is easier to compute, e.g. because it factorizes over variables.

The parameters ?? of the true generative distribution p (i.e. the model parameters) can be optimized jointly with the parameters ?? of the approximate distribution q by maximizing the evidence lower bound (ELBO).

BID3 combine a locally linear gaussian state space model (LGSSM) with a variational autoencoder BID16 BID20 ) that learns to encode highdimensional sensory input data into a low-dimensional latent representation in an unsupervised way.

This latent encoding is used as observations to the LGSSM.

In addition, an LSTM network predicts the parameters of the process model from the history of latent encodings.

Watter et al. (2015) follow a similar approach.

Their method predicts a latent representation as well as the parameters of a a locally linear gaussian process model from the observations using variational autoencoders.

A regularization term enforces that the predicted and inferred representation match for each timestep.

In contrast to the previous works, where the encoding from observations into latent space is learned directly, BID13 train a variational autoencoder to only predict the parameters of the process model and the process noise from the observations.

This enforces that the learned latent state contains all information necessary to predict the next state, without relying on the observations at the next timestep.

All of the methods discussed here focus on unsupervised learning of observation and process models in systems with unknown state representation.

In contrast, in our work we leverage prior knowledge about the process model and the state representation obtained from first-order principles.

This enables supervised learning and is thought to improve the generalization ability of the learned parts BID17 .

Unsupervised training by backpropagation through filtering algorithms is possible as well, as was demonstrated in BID8 .The main conceptual difference between variational methods and learning in differentiable filters is that variational methods learn to perform inference in state space models by optimizing an approximate posterior distribution.

Bayesian filters, on the other hand, provide fixed algorithms for approximating the posterior, which have been shown to work well in practice for many problems.

Using these algorithmic priors intuitively makes learning in differentiable filters easier, but restricts the class of models that can be learned.

Variational methods solve a more difficult learning problem, but can fit the training data more freely.

How big this difference really is, however, depends on how the approximate posterior and the generative model in the variational approach are structured.

An in-depth analysis of the effects of the two approaches on training and the learned models has, to our knowledge, not yet been attempted and would be an interesting direction for future work.

We implement the filtering methods presented in Section 2 as recurrent neural networks in tensorflow BID0 .

In this section, we describe how the learnable noise models are parametrized and used in the filters.

For more details about the implementation please refer to the Appendix 7.1.

In state space models, the observation model is a generative model that predicts observations from the state z t = h(x).

In practice, it is however often hard to find such a model that directly predicts the potentially high-dimensional raw sensory signals without making strong assumptions.

We therefore use the method proposed by BID6 and train a discriminative neural network o with parameters w o to preprocess the raw sensory data D and thus create a more compact representation of the observations z = o(D, w o ).In our experiments, the perception network directly extracts some of the components of the state x from D, such that the actual observation model h becomes a simple selection operation.

Besides from z, the perception networks can also predict the diagonal entries of the observation noise covariance matrix R.We pretrain the perception networks for all experiments to predict z, but not R. Since z is a subset of the components of x, this requires no additional data annotation.

In experiments where R is learned, we initialize the prediction to reasonable values using a trainable bias, otherwise we use a fixed diagonal matrix as R.

For learning the process noise, we consider two different conditions: constant and heteroscedastic.

In all cases, we assume that the process noise at time t can be described by a zero-mean Gaussian distribution with diagonal covariance matrix Q t .

The constant noise model consists of one trainable variable w q that represents the diagonal entries of Q.In the heteroscedastic case, the diagonal elements are predicted from the current state x t and (if available) the control input u t , by a 3-layer MLP g with weights w g : diag(Q) = g(x t , u t , w g ).

In the UKF and MCUKF, we predict a separate Q i for every sigma point and then compute Q as their weighted mean.

In all variants of the Kalman Filter, the process noise enters the prediction step in the update of the covariance matrix ?? (Equations 2, 12) and influences the update step through the Kalman Gain (Equation 4).

In the Particle Filter, it is used for sampling particles from the process model (Equation 13).

Following BID9 , we implement this step with the reparametrization trick BID16 : DISPLAYFORM0 4.3 TRAININGWe train the noise models end-to-end trough the filters using the Adam optimizer BID15 and backpropagation through time.

The loss consists of three components, (i) the negative log likelihood of the true state given the believe, (ii) the Euclidean error between the ground truth state and the predicted mean and (iii) a regularization term on the weights of the trainable noise models.

DISPLAYFORM1 Here l 0...T is the ground truth state sequence, ?? 0...T and ?? 0...T denote the sequence of prediction mean and covariance respectively.

w contains the weights of the trainable noise models (which influence the prediction of ?? and ??) like w o or w g .

The ?? i are scaling factors that can be chosen dependent on the magnitude of the loss components.

The likelihood loss encourages the network to predict noise values that minimize the overall predicted covariance (i.e. the uncertainty about the predicted state) while at the same time penalizing high confidence predictions with large errors.

In practice, we found that during learning, the models often optimized the likelihood by only increasing the predicted variance instead of minimizing the prediction error.

Therefore, we added the second component of the loss to enforce low overall prediction errors.

Both the MCUKF and the Particle Filter approximate the state by sampling and require a potentially large number of sigma points/particles for accurate prediction.

During training, we have to limit the number of samples to 100, as memory consumption and computation time increase with the number of samples.

For testing, we can use much higher numbers of particles/sigma points.

It has been shown before that using the algorithmic structure of Bayesian Filters to enable end-toend learning is very beneficial for learning the process and observation models of the filters.

Here we evaluate how end-to-end learning of heteroscedastic noise models affects the performance of the different filtering algorithms.

These noise models quantify the accuracy of the process and observation models.

For this, we test each filter under five conditions: Without learning, only learning the observation noise R, learning only heteroscedastic process noise Q h and learning both with constant or heteroscedastic process noise (R + Q, R + Q h ).

As the influence of modeling the noise can depend on the task, we perform experiments on two different applications.

As a first application we chose the Kitti Visual Odometry task BID4 ) that was also evaluated in BID6 and BID9 .

The aim is to estimate the position and orientation of a driving car given a sequence of rgb images from a front facing camera and the true initial state.

The state is 5-dimensional and includes the position p and orientation ?? of the car as well as the current linear and angular velocity v and??.

As the control inputs are unknown, the estimated velocities are predicted by sampling random accelerations a,??, according to the process noise for v and??.

The position and heading estimate are update by Euler integration (see Appendix 7.2.1).While the dynamics model is simple, the challenge comes from the fact that the drivers actions are not known and the absolute position and orientation are not observable.

The filters can therefore only rely on estimating the angular and linear velocity from pairs of input images to update the state, but the uncertainty about the position and heading will inevitably grow due to missing feedback.

We pretrain a neural network to extract this information from the current input image and the difference image between the current and previous one.

The network architecture is the same as was used in BID6 BID9 , we only replace the response normalization layers with tensorflow's standard batch normalization layers.

Since both related work allowed for finetuning of the perception network trough the filter, we do the same here for better comparability of results.

As in BID9 , we test the Particle Filter using 1000 particles and also use 1000 sigma points for the MCUKF.The process and observation noise are initialized to the same values in every condition.

For the observation noise, we look at the average error of the perception network at the end of the pretraining phase.

To set the process noise, we use the ground truth standard deviation of the velocities to initialize the terms for linear and angular velocity.

The terms for position and heading are initialized to identity.

See the Appendix 7.2.1 for exact values.

The Kitti Visual Odometry dataset consists 11 trajectories of varying length (from 270 to over 4500 steps) with ground truth annotations for position and heading and image sequences from two different cameras collected at 10 Hz.

We use the two shortest sequences for validation and perform a 9-fold cross-validation on the remaining sequences.

We use both image sequences from each trajectory and further augment the data by adding the mirrored sequences as well.

For training, we extract non-overlapping sequences of length 50 with a different random starting point for each image-sequence.

The sequences for validation and testing consist of 100 timesteps.

No learning, learning constant observation noise R, learning heteroscedastic process noise Q h , learning constant observation and process noise R + Q, learning constant observation noise and heteroscedastic process noise R + Q h .

In each condition, the perception network was pretrained offline and finetuned through the filters.

We evaluate the models on different trajectories with 100 timesteps.

As in BID9 BID6 we report mean and std of the end-point-error in position and orientation normalized by the distance between start and end point.

TAB0 contains the average normalized end-point-errors for the different filters and noise learning conditions.

On this task, the EKF outperforms the other filters even without learning the noise models and does not gain a lot from leaning them.

The Particle Filter as well as the MCUKF perform badly without learning or when training the observation noise R alone.

While learning a constant process noise Q improved their results, learning a heteroscedastic process noise model lead to much bigger improvements for the MCUKF and for the PF when predicting the heading of the car.

This does not necessarily mean that the task follows a heteroscedastic noise model, especially since the EKF and UKF do not show big differences between constant and heteroscedastic noise.

Instead, it seems like the heteroscedastic process noise model facilitates the training process: When the process noise is trained with a heteroscedastic process noise model, we observe that it quickly converges towards zero for position and orientation, which is the best choice for this task.

In the constant noise setting, this convergence is much slower and the models do not fully converge during the training. ).

This could be due to differences in the implementation of the observation model BID9 use a model that directly predicts the likelihood of each particle instead of a distribution over velocities), the different initial values for the process noise or the soft resampling we use (see Appendix 7.1.3).For this particular task, the MCUKF turns out to be a bad choice: Without learning a suitable process noise model, it mostly fails to predict any movement of the car.

This is caused by high uncertainty about the orientation of the car, both due to the bad initialization of the process noise and the accumulating uncertainty during tracking: If the sampled sigma points are too different in estimated orientation, their movement cancels each other out when calculating the mean.

The standard UKF performs better, because of the symmetry in the sigma point construction (see Eq. 11) and because it keeps the previous mean as a sigma point that is weighted higher than the remaining points and thus enforces movement in the correct direction.

In general, it is not surprising that the EKF performs best on this task: First, the process model is smooth and not highly non-linear, such that the EKF provides a good approximation of the posterior.

Second, the main difference to the other filters is that the PF and the UKF variants generate additional uncertainty about the position and heading of the car by sampling particles or constructing sigma points.

This uncertainty would usually be resolved by observations, such that more weight can be given to the particles or sigma points that are closer to the true state.

In visual odometry, however, there are no observations of heading and position and there is thus nothing to gain from exploring values that deviate from the estimated mean.

In the visual odometry problem, the main challenges were perception and dealing with the inevitably increasing uncertainty.

Our second experiment in contrast addresses a task with more complex dynamics: quasi-static planar pushing.

Apart from having non-linear and discontinuous dynamics (when the pusher makes or breaks contact with the object), BID1 also showed that the noise in the system can be best captured by a heteroscedastic noise model.

The state we try to estimate has 10 dimensions: the 2d position p and orientation ?? of the object, two friction-related parameters l and m, the 2d contact point between pusher and object r and the normal to the object's surface there n as well as a variable s that indicates if the pusher is in contact with the object or not.

For predicting the next state, we use an analytical model of quasi-static planar pushing BID18 BID17 .

It predicts the linear and angular velocity of the object (v, ??) given the pusher velocity u and the current state.

Details can be found in the Appendix 7.2.2.We use coordinate images (like a depth image, but with all 3 coordinates as channels) of the scene at time t ??? 1 and t as input, and train a neural network to extract the position of the object, the contact point and normal as well as if the pusher is in contact with the object or not.

Besides from the friction-related parameters, the orientation of the object, ?? t , is the only state component that cannot be estimated directly from the input images.

As absolute orientation of an object is not defined (without giving a reference for each object), we cannot extract it from the images.

Instead, we train the network to observe the change in orientation ?? between the two images (up to symmetries).In contrast to the visual odometry task in the previous experiment, we do not assume that the initial state is correct.

All models are thus evaluated on five different initial conditions with varying error and we report the average error and standard deviation across these five setting.

We also do not finetune the perception model in this experiment.

We again use 100 sigma points or particles during training for the MCUKF and PF.

During test-time, the particle filter uses 1000 particles while we limit the MCUKF to 500 sigma points.

The MIT Push dataset (Yu et al., 2016) consist of more than a million real robot push sequences to eleven different objects on four different surface materials.

For each sequence the original dataset contains the object position, the position of the pusher as well as force recordings.

We use the tools described by BID17 to get additional annotations for the remaining state components and for rendering depth images.

In contrast to BID17 our images also show the robot arm and are taken from a more realistic camera angle.

We use data from pushes with a velocity of 50 mm s and render images with a frequency of 18 Hz.

This results in very short sequences of about 15 images per push.

We extend these sequences to 100 steps by chaining multiple pushes and adding in between pusher movement when necessary.

We use subsequences of ten steps for training and the full 100 steps for testing.

Without learning In the first two columns of Table 2 , we compare the tracking performance of the different filters without learning any of the noise models.

For the first column, we set the diagonal values of Q to 0.01 and those of R to 100 such that the filters place too high confidence in the process model and too low confidence in the observations.

In the second condition, we used the average prediction error of the analytical model and the preprocessing network on the ground truth data to set Q and R to realistic values.

EKF 11.8 ?? 0.54 3.9 ?? 0.02 3.7 ?? 0.01 3.9 ?? 0.01 3.8 ?? 0.01 3.9 ?? 0.02 UKF 9.3 ?? 0.31 3.8 ?? 0.01 3.8 ?? 0.02 3.8 ?? 0.02 3.8 ?? 0.01 3.9 ?? 0.003 MCUKF 9.2 ?? 0.33 3.8 ?? 0.01 3.7 ?? 0.01 3.8 ?? 0.01 3.7 ?? 0.01 3.8 ?? 0.01 PF 56.5 ?? 0.11 7.4 ?? 0.30 20.2 ?? 0.62 3.3 ?? 0.23 7.0 ?? 0.21 3.0 ?? 0.20 DISPLAYFORM0 EKF 18.9 ?? 0.57 9.3 ?? 0.3 9.9 ?? 0.21 9.4 ?? 0.39 8.4 ?? 0.24 9.3 ?? 0.33 UKF 20.6 ?? 1.11 9.4 ?? 0.28 10.8 ?? 0.22 9.34 ?? 0.26 9.5 ?? 0.17 6.1 ?? 0.14 MCUKF 21.4 ?? 1.4 10.1 ?? 0.43 9.5 ?? 0.38 7.6 ?? 0.26 8.4 ?? 0.2 6.5 ?? 0.21 PF 28.4 ?? 0.07 21.1 ?? 1.1 16.4 ?? 0.46 8.8 ?? 0.18 12.2 ?? 0.45 10.1 ?? 0.37 Table 2 : Planar Pushing task.

Evaluation of four non-linear filters under five different noise learning conditions: No learning 1 (with unrealistic noise), No learning 2 (with realistic noise) , learning constant observation noise R, learning heteroscedastic process noise Q h , learning constant observation and process noise R + Q, learning constant observation noise and heteroscedastic process noise R + Q h .

Mean and standard deviation of tracking errors on the planar pushing task averaged over five different initial conditions.

Tracking errors are mean squared error in position and orientation of the object averaged over all timesteps in the sequence.

While all filters perform worse on the unrealistic noise setting, the Particle Filter is affected the most.

This is presumably because without well-tuned noise models, it samples many particles far away from the true state and cannot discriminate well between likely and unlikely particles given the observations.

Learning the noise models The remaining columns of Table 2 show the results when learning the different combinations of noise models.

The process and observation noise are initialized to the realistic values from the no learning setting for every condition.

We can see that the performance of the Extended Kalman Filter again remains mostly constant over all conditions and also does not improve much over the model with well-tuned noise.

Both the UKF and the MCUKF do not show much difference for tracking the position of the object.

We see a slightly improved performance for tracking the orientation of the object when a constant process noise model is trained and a stronger improvement with the heteroscedastic Q h .

This is consistent with the results in the previous experiment, as the orientation of the object can again not be observed directly and it is thus not desirable to vary it much when creating the sigma points.

Overall, the traditional UKF with trained R and heteroscedastic Q performs best, but the MCUKF is similar and could potentially perform better if more sigma points were sampled.

In this experiment, the Particle Filter profits most from learning: In the two conditions with heteroscedastic process noise, its tracking performance improves dramatically and even outperforms the other filters on the position metric.

The improvement over the untrained setting is much smaller when Q is constrained to be constant.

Why is learning a heteroscedastic process noise model so important for the PF?

We believe that learning a separate Q for each particle helps the filter to steer the particle set towards more likely regions of the state space.

It can for example get rid of particles that encode a state configuration that is not physically plausible and will therefore lead to a bad prediction from the analytical model by sampling higher noise and thus decreasing the likelihood of the particle.

Training the observation noise R did not have a very big effect in this experiment, but inspecting the learned diagonal values showed that all filters learned to predict higher uncertainty for the y coordinate of positions, which makes sense as the y axis of the world frame points towards the background of the image and perspective transform thus reduces the accuracy in this direction.

In contrast to the results in BID6 , we did not see any evidence that the heterostochasticity of the observation noise was helpful.

This can probably be explained by the absence of complete occlusions of the object in our dataset.

We could also not identify any other common feature of scenes for which our prediction model produced high prediction errors.

It is therefore likely that a constant observation noise model would have been sufficient in this setting.

We proposed to optimize the process and observation noise for Bayesian Filters through end-to-end training and evaluated the method with different filtering algorithms and on two robotic applications.

Our experiments showed that learning the process noise is especially important for filters that sample around the mean estimate of the state, like the Particle Filter but also the Unscented Kalman Filters.

The Extended Kalman Filter in contrast proved to be most robust to suboptimal choices of the noise models.

While this makes it a good choice for problems with simple and smooth dynamics, our experiments on the pushing task demonstrated that the (optimized) Unscented Filters can perform better on problems with more complex and even discontinuous dynamics.

Training a state-dependent process noise model instead of a constant one improves the prediction accuracy for dynamic systems that are expected to have heteroscedastic noise.

In our experiments, it also facilitated learning in general and lead to faster convergence of the models.

We also used a heteroscedastic observation noise model in all our experiments.

But different from the results in BID6 , we could not see a large benefit from it: Inspection on the pushing task showed that larger errors in the prediction of the preprocessing networks were not associated with higher observation noise.

Identifying inputs that will lead to bad predictions is a difficult task if no obvious problems like occlusions are present to explain such outliers.

Developing better methods for communicating uncertainty about the predictions of a neural network would thus be an impotent next step to further improve the performance of differentiable Bayesian Filters.

The basic steps of the Extended Kalman Filter can be directly implemented in Tensorflow without any modifications.

The only aspect of interest is how to compute the Jacobians of the process and observation model.

Tensorflow implements auto differentiation, but has (as of now) no native support for computing Jacobians.

While it can be done, it requires looping over the dimensions of the differentiated variable one by one, which we found to be relatively slow, especially during graph-construction.

We therefore recommend to manually derive the Jacobians where applicable.

Like for the EKF, implementing the prediction and update step of the UKF in tensorflow is straight forward.

For constructing the sigma points, it is necessary to compute the matrix square root of the estimated covariance ??. This is commonly done using the Cholesky Decomposition, which is also available in tensorflow.

In practice, the Cholesky decomposition however often failed.

Instead, we used the more robust singular value decomposition.

For the MCUKF, we sample the sigma points from a Gaussian distribution with the same mean and covariance as the current estimate using tensorflow's distribution tools.

Internally, this also relies on the Cholesky decomposition and thus requires ?? to be positive semidefinite at all times.7.1.3 PF Our particle Filter implementation is very similar to the variant proposed by BID9 that is available online 1 .

We combine it with the differentiable resampling technique proposed by BID12 to enable backpropagation through the weights.

Another difference is that we do not train a network to directly predict the likelihood of an observation given a particle.

Instead, we use the same preprocessing network as for the other filtering types, which outputs the observations z and the estimated covariance matrix of the observation model R. Given these, we compute the probability of z under a gaussian distribution defined by the predicted observations for each particle and R. This approach might be more challenging to train (as the likelihoods become very small if the observation noise is too low) but allows for a better comparison with the other filters.

A particular difficulty in training differentiable filters in tensorflow is to ensure that the estimated covariance matrices are positive semidefinite at any time, even if the filters diverge.

This ensures for example that they can be inverted for computing likelihoods or the Kalman Gain, which will otherwise result in an error that stops the training.

We employ the method described in BID7 to reset the covariance matrices to the nearest positive semidefinite matrix after every iteration.

The process model for the visual odometry task is defined as p x p y t = p x p y t???1 + ???tv t???1 sin(?? t???1 ) cos(?? t???1 ) v t = v t???1 + ???ta t ?? t = ?? t???1 + ???t?? t???1??t =?? t???1 + ???t?? tFor the architecture of the preprocessing network, we refer to BID6 .

We initialize the process noise with diagonal values of diag(Q) = (1 1 1 11.

0.0225) and the observation noise with diag(R) = (4. 1)

Data FIG2 shows two examples of the rendered images we use in the pushing task.

While we actually use coordinate images as input, we show rgb images here for better visibility.

Preprocessing Network The architecture of the preprocessing network that infers z from the raw input images is shown in FIG3 .

It is similar to the network described in BID17 for inferring object position p, contact point r, contact normal n and the contact indicator s from the scene.

We add the left part that computes ??, the difference in object rotation between the current and the previous image.

For this, we extract patches around the predicted object position in both images and feed both into a convolutional and fully-connected network to infer ??.

Process Model Given the output of the analytical model (v t , ?? t ), we formulate the process model f (x t , u t ) as p t+1 = p t + v t r t+1 = r t + u t ?? t+1 = ?? t + ?? t n t+1 = R(?? t )n t l t+1 = l t s t+1 = s t m t+1 = m t Here, we make the simplifying assumption that the pusher will not make or break contact and that s is thus constant.

To predict the next contact point, we update it with the movement of the pusher.

The accuracy of this prediction is bounded by the radius of the pusher, which is rather small in our case.

For predicting the next normal at the contact point, we assume that the position of the contact point on the object does not change and the normal thus remains constant in the object coordinate frame.

Given this assumption, the only thing we need to do is to adapt the orientation of the normal to the rotation of the object, where R(?? t ) denotes a rotation matrix that rotates n by ?? t .

@highlight

We evaluate learning heteroscedastic noise models within different Differentiable Bayes Filters

@highlight

Proposes to learn heteroscedastic noise models from data by optimizing the prediction likelihood end-toend through differentiable Bayesian Filters and two different versions of the Unscented Kalman Filter

@highlight

Revisits Bayes filters and evaluates the benefit of training the observation and process noise models while keeping all other models fixed

@highlight

This paper presents a method to learn and use state and observation dependent noise in traditional Bayesian filtering algorithms. The approach consists of constructing a neural network model which takes as input the raw observation data and produces a compact representation and an associated diagonal covariance.