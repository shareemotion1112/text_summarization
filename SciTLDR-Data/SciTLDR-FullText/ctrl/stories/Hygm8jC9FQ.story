A state-of-the-art generative model, a ”factorized action variational autoencoder (FAVAE),” is presented for learning disentangled and interpretable representations from sequential data via the information bottleneck without supervision.

The purpose of disentangled representation learning is to obtain interpretable and transferable representations from data.

We focused on the disentangled representation of sequential data because there is a wide range of potential applications if disentanglement representation is extended to sequential data such as video, speech, and stock price data.

Sequential data is characterized by dynamic factors and static factors: dynamic factors are time-dependent, and static factors are independent of time.

Previous works succeed in disentangling static factors and dynamic factors by explicitly modeling the priors of latent variables to distinguish between static and dynamic factors.

However, this model can not disentangle representations between dynamic factors, such as disentangling ”picking” and ”throwing” in robotic tasks.

In this paper, we propose new model that can disentangle multiple dynamic factors.

Since our method does not require modeling priors, it is capable of disentangling ”between” dynamic factors.

In experiments, we show that FAVAE can extract the disentangled dynamic factors.

Representation learning is one of the most fundamental problems in machine learning.

A real world data distribution can be regarded as a low-dimensional manifold in a high-dimensional space BID3 .

Generative models in deep learning, such as the variational autoencoder (VAE) BID25 and the generative adversarial network (GAN) BID15 , are able to learn low-dimensional manifold representation (factor) as a latent variable.

The factors are fundamental components such as position, color, and degree of smiling in an image of a human face BID27 .

Disentangled representation is defined as a single factor being represented by a single latent variable BID3 .

Thus, if in a model of learned disentangled representation, shifting one latent variable while leaving the others fixed generates data showing that only the corresponding factor was changed.

This is called latent traversals (a good demonstration of which was given by BID17 1 ).

There are two advantages of disentangled representation.

First, latent variables are interpretable.

Second, the disentangled representation is generalizable and robust against adversarial attacks BID1 .We focus on the disentangled representation learning of sequential data.

Sequential data is characterized by dynamic factors and static factors: dynamic factors are time dependent, and static factors are independent of time.

With disentangled representation learning from sequential data, we should be able to extract dynamic factors that cannot be extracted by disentangled representation learning models for non-sequential data such as β-VAE BID17 b) and InfoGAN BID8 .

The concept of disentangled representation learning for sequential data is illustrated in Fig. 1 .

Consider that the pseudo-dataset of the movement of a submarine has a dynamic factor: the trajectory shape.

The disentangled representation learning model for sequential data can extract this shape.

On the other hand, since the disentangled representation learning model for non-sequential data does not consider the sequence of data, it merely extracts the x-position and y-position.

Figure 1: Illustration of how FAVAE differs from β-VAE.

β-VAE does not accept data sequentially; it cannot differentiate data points from different trajectories or sequences of data points.

FAVAE considers a sequence of data points, taking all data points in a trajectory as one datum.

For example, for a pseudo-dataset representing the trajectory of a submarine (1a,1c), β-VAE accepts 11 different positions of the submarine as non-sequential data while FAVAE accepts three different trajectories of the submarine as sequential data.

Therefore, the latent variable in β-VAE learns only the coordinates of the submarine, and the latent traversal shows the change in the submarines position.

On the other hand, FAVAE learns the factor that controls the trajectory of the submarine, so the latent traversal shows the change in the submarines trajectory.

There is a wide range of potential applications if we extend disentanglement representation to sequential data such as speech, video, and stock market data.

For example, disentangled representation learning for stock price data can extract the fundamental trend of a given stock price.

Another application is the reduction of action space in reinforcement learning.

Extracting dynamic factors would enable the generation of macro-actions BID11 , which are sets of sequential actions that represent the fundamental factors of the actions.

Thus, disentangled representation learning for sequential data opens the door to new areas of research.

Very recent related work BID22 BID26 ) separated factors of sequential data into dynamic and static factors.

The factorized hierarchical variational autoencoder (FHVAE) BID22 ) is based on a graphical model using latent variables with different time dependencies.

By maximizing the variational lower bound of the graphical model, the FHVAE separates the different time dependent factors such as the dynamic and static factors.

The VAE architecture developed by BID26 is the same as the FHVAE in terms of the time dependencies of the latent variables.

Since these models require different time dependencies for the latent variables, these approaches cannot be used disentangle variables with the same time dependency factor.

We address this problem by taking a different approach.

First, we analyze the root cause of disentanglement from the perspective of information theory.

As a result, the term causing disentanglement is derived from a more fundamental rule: reduce the mutual dependence between the input and output of an encoder while keeping the reconstruction of the data.

This is called the information bottleneck (IB) principle.

We naturally extend this principle to sequential data from the relationship between x and z to x t:T and z. This enables the separation of multiple dynamic factors as a consequence of information compression.

It is difficult to learn a disentangled representation of sequential data since not only the feature space but also the time space should be compressed.

We created the factorized action variational autoencoder (FAVAE) in which we implemented the concept of information capacity to stabilize learning and a ladder network to learn a disentangled representation in accordance with the level of data abstraction.

Since our model is a more general model without the restriction of a graphical model design to distinguish between static and dynamic factors, it can separate depen-dency factors occurring at the same time.

Moreover, it can separate factors into dynamic and static factors.2 DISENTANGLEMENT FOR NON-SEQUENTIAL DATA β-VAE BID17 b) is a commonly used method for learning disentangled representations based on the VAE framework BID25 ) for a generative model.

The VAE can estimate the probability density from data x. The objective function of the VAE maximizes the evidence lower bound (ELBO) of log p (x) as DISPLAYFORM0 where z is latent variable, D KL is the Kullback-Leibler divergence, and q (z|x) is an approximated distribution of p (z|x).

D KL (q (z|x) ||p (z|x)) reduces to zero as the ELBO L VAE increases; thus, q (z|x) learns a good approximation of p (z|x).

The ELBO is defined as DISPLAYFORM1 where the first term, E q(z|x) [log p (x|z)], is a reconstruction term used to reconstruct x, and the second term D KL (q (z|x) ||p (z)) is a regularization term used to regularize posterior q (z|x).

Encoder q (z|x) and decoder p (x|z) are learned in the VAE.Next we will explain how β-VAE extracts disentangled representations from unlabeled data.

β-VAE is an extension of the coefficient β > 1 of the regularization term DISPLAYFORM2 where β > 1 and p (z) = N (0, 1).

β-VAE promotes disentangled representation learning via the Kullback-Leibler divergence term.

As β increases, the latent variable q (z|x) approaches the prior p (z) ; therefore, each z i is pressured to learn the probability distribution of N (0, 1).

However, if all latent variables z i become N (0, 1), the model cannot reconstruct x. As a result, as long as z reconstructs x, β-VAE reduces the information of z.

To clarify the origin of disentanglement, we will explain the regularization term.

The regularization term has been decomposed into three terms BID7 BID23 BID21 : DISPLAYFORM0 where z j denotes the j-th dimension of the latent variable.

The second term, which is called "total correlation" in information theory, quantifies the redundancy or dependency among a set of n random variables BID31 .

β-TCVAE has been experimentally shown to reduce the total correlation causing disentanglement BID7 .

The third term indirectly causes disentanglement by bringing q (z|x) close to the independent standard normal distribution p (z).

The first term is mutual information between the data variable and latent variable based on the empirical data distribution.

Minimizing the regularization term causes disentanglement but disturbs reconstruction via the first term in Eq. (4).

The shift C scheme was proposed BID5 as a means to solve this conflict: DISPLAYFORM1 where constant shift C, which is called "information capacity," linearly increases during training.

This shift C can be understood from the point of view of an information bottleneck BID30 .

The VAE can be derived by maximizing the ELBO, but β-VAE can no longer be interpreted as an ELBO once this scheme has been applied.

The objective function of β-VAE is derived from the information bottleneck BID1 BID0 BID30 BID6 .

DISPLAYFORM2 where C is the information capacity andx is the empirical distribution.

Solving this equation by using Lagrange multipliers drives the objective function of β-VAE Eq. FORMULA4 with β as the Lagrange multiplier (details in Appendix B of BID1 ).

In Eq. FORMULA4 , information capacity C prevents I (x, z) from becoming zero.

In the information bottleneck literature, y typically stands for a classification task; however, the formulation can be related to the autoencoding objective BID1 .

Therefore, the objective function of β-VAE can be understood using the information bottleneck principle.

Our proposed FAVAE model learns disentangled and interpretable representations from sequential data without supervision.

We consider sequential data x 1:T ≡ {x 1 , x 2 , · · · , x T } generated from a latent variable model, DISPLAYFORM0 For sequential data, we replace x with (x 1:T ) in Eq. 5.

The objective function of the FAVAE model is DISPLAYFORM1 where p (z) = N (0, 1).

The variational recurrent neural network BID10 and stochastic recurrent neural network BID13 ) extend the VAE model to a recurrent framework.

The priors of both networks are dependent on time.

The time dependent prior experimentally improves the ELBO.

In contrast, the prior of our model is independent of time like those of the stochastic recurrent network BID2 and the Deep Recurrent Attentive Writer (DRAW) neural network architecture BID16 ; this is because FAVAE is disentangled representation learning rather than density estimation.

For better understanding, consider FAVAE from the perspective of IB.

As with β-VAE, FAVAE can be understood from the information bottleneck principle.

DISPLAYFORM2 wherex 1:T follows an empirical distribution.

These principles make the representation of z compact while reconstruction of the sequential data is represented by x 1:T (see Appendix A).

Figure 2: FAVAE architecture.

An important extension to FAVAE is a hierarchical representation scheme inspired by the VLAE BID32 .

Encoder q (z|x 1:T ) within a ladder network is defined as DISPLAYFORM0 where l is a layer index, h 0 ≡ x 1:T , and f is a time convolution network, which is explained in the next section.

Decoder p (x 1:T |z) within the ladder network is defined as DISPLAYFORM1 DISPLAYFORM2 where g l is the time deconvolution network with l = 1, · · · , L − 1, and r is a distribution family parameterized by g 0 (z 0 ).

The gate computes the Hadamard product of its learnable parameter and DISPLAYFORM3 (e) FAVAE, 1st z in 1st ladder.

Figure 3: Visualization of latent traversal of β-VAE and FAVAE.

On one sampled trajectory (red), each latent variable is traversed and purple and/or blue points are generated.

The color corresponds to the value of the traversed latent variable.

3a represents all data trajectories of 2D reaching.input tensor.

We set r as a fixed-variance factored Gaussian distribution with the mean given by µ t:T = g 0 (z 0 ).

Fig. (2) shows the architecture of the proposed model.

The difference between each ladder network in the model is the number of convolution networks through which data passes.

The abstract expressions should differ between ladders since the time convolution layer abstracts sequential data.

Without the ladder network, the proposed method can disentangle only the representations at the same level of abstraction; with the ladder network, it can disentangle representations at different levels of abstraction.

There are several mainstream neural network models designed for sequential data, such as the long short-term memory model BID20 , the gated recurrent unit model BID9 , and the quasi-recurrent neural network QRNN BID4 .

However, the VLAE has a hierarchical structure created by abstracting a convolutional neural network, so it is simple to add the time convolution of the QRNN to our model.

The input data are x t,i , where t is the time index and i is the dimension of the feature vector index.

The time convolution considers the dimensions of feature vector j as a convolution channel and performs convolution in the time direction: DISPLAYFORM0 where j is the channel index.

The proposed FAVAE model has a network similar to the VAE one regarding time convolution and a loss function similar to the β-VAE one (Eq. FORMULA7 ).

We used the batch normalization BID19 and ReLU as activation functions though other variations are possible.

For example, 1d convolutional neural networks use a filter size of 3 and a stride of 2 and do not use a pooling layer.

While latent traversals are useful for checking the success or failure of disentanglement, quantification of the disentanglement is required for reliably evaluating the model.

Various disentanglement quantification methods have been reported BID12 BID7 BID23 BID18 a) , but there is no standard method.

We use the mutual information gap (MIG) BID7 as the metric for disentanglement.

The basic idea of MIG is measuring the mutual information between latent variables z j and a ground truth factor v k .

Higher mutual information means that z j contains more information regarding v k .

DISPLAYFORM0 and H (v k ) is entropy for normalization.

In our evaluation we experimentally measure disentanglement with MIG.

6 RELATED WORKSeveral recently reported models BID22 BID26 graphically disentangle static and dynamic factors in sequential data such as speech data and video data BID14 BID28 .

In contrast, our model performs disentanglement by using a loss function (see Eq. 8).

The advantage of the graphical models is that they can control the interpretable factors by controlling the priors time dependency.

Since dynamic factors have the same time dependency, these models cannot disentangle dynamic factors.

A loss function model can disentangle sets of dynamic factors as well as disentangle static and dynamic factors.

We evaluated our model experimentally using three sequential datasets: 2D Reaching, 2D Wavy Reaching, and Gripper.

We used a batch size of 128 and the Adam (Kingma & Ba, 2014) optimizer with a learning rate of 10 −3 .

To determine the differences between FAVAE and β-VAE, we used a bi-dimensional space reaching dataset.

Starting from point (0, 0), the point travels to goal position (-0.1, +1) or (+0.1, +1).

There are ten possible trajectories to each goal; five are curved inward, and the other five are curved outward.

The degree of curvature for all five trajectories is different.

The number of factor combinations was thus 20 (2x2x5).

The trajectory length was 1000, so the size of one trajectory was [1000x2].We compared the performances of β-VAE and FAVAE trained on the 2D Reaching dataset.

The results of latent traversal are transforming one dimension of latent variable z into another value and reconstructing something from the traversed latent variables.

β-VAE, which is only able to learn from every point of a trajectory separately, encodes data points into latent variables that are parallel to the x and y axes (3b, 3c).

In contrast, FAVAE learns through one entire trajectory and can encode disentangled representation effectively so that feasible trajectories are generated from traversed latent variables (3d, 3e).

To confirm the effect of disentanglement through information bottleneck, we evaluated the validity of our model under more complex factors by adding more factors to the 2D Reaching dataset.

Five factors in total generated data compared to the three factors that generate data in 2D Reaching.

This modified dataset differed in that four out of the five factors affect only part of the trajectory: two of them affect the first half, and the other two affect the second half.

This means that the model should be able to focus on a certain part of the whole trajectory and be able to extract factors related to that part.

A detailed explanation of these factors is given in Appendix B.We compared various models on the basis of MIG to demonstrate the validity of our proposed model in comparison of a time convolution AE in which a loss function is used only for the autoencoder (β = 0), FAVAE without the ladder network and information capacity C, and FAVAE with the ladder network and information capacity C. As shown in TAB0 , FAVAE with the ladder network and C had the highest MIG scores for 2D Reaching and 2D Wavy Reaching.

This indicates that this model learned a disentangled representation best.

Note that for 2D Reaching, the best value for C was small, meaning that there was little effect from adding C (since the dataset was simple, this task can be solved even if the amount of information of z is small). .

When C was not used, the model could not reconstruct data when β was high; thus, disentangled representation was not learned well when β was high.

When C was used, the MIG score increased with β while reconstruction loss was suppressed.

The latent traversal results for 2D Wavy Reaching are plotted in FIG2 .

Even though not all learned representations are perfectly disentangled, the visualization shows that all five generation factors were learned from five latent variables; the other latent variables did not learn any meaningful factors, indicating that the factors could be expressed as a combination of five "active" latent variables.

We tested our model for β = 300.

The use of a ladder network in our model improved disentangled representation learning and minimized the reconstruction loss.

The graph in Fig. 6 shows the MIG scores for networks with different numbers of ladders.

The error bars represent the standard deviation for ten repetitive trials.

Using all three ladders resulted in the minimum reconstruction loss with the highest MIG score (green curve) except for "Higher Ladder One".

"

Higher Ladder One" has a large reconstruction error.

To evaluate the effectiveness of the video dataset, we trained our model with the Sprites dataset which is used in BID26 .

This dataset has sequential length = 8 RGB video data with 3 × 64 × 64.

This data set consists of static factor and dynamic factor.

We note that the motion is not created with the combination of dynamic factors, and each motion exists individually (detail is E.2).

TAB3 show the factors used in our experiment.

We executed disentangled representation learning by using the FAVAE model with β = 20, C = [0.3, 0.17, 0.06] and network architecture used for this training is explained in Section F.1.

Fig. 7 shows the results of latent traversal, and we chose two z values to change from z = 3 to 3.

Since this dataset is composed of discrete factors, we show two z values at a time.

The latent variables in the 1st ladder extract expressions of motion (4th z in 1st ladder), pants color (5th z in 1st ladder), direction of character (6th z in 1st ladder) and the shirts color (7th z in 1st ladder).

The latent variables in the 2nd ladder extract expressions of the hair color (1st z in 2nd ladder) and the body color (2nd z in 2nd ladder).

FHVAE can extract the disentangled representations between static factors and dynamic factor in high dimension dataset.

Our factorized action variational autoencoder (FAVAVE) generative model learns disentangled and interpretable representations via the information bottleneck from sequential data.

Evaluation using three sequential datasets demonstrated that it can learn disentangled representations.

Future work includes extending the time convolution part to a sequence-to-sequence model BID29 and applying the model to actions of reinforcement learning to reduce the pattern of actions.

A INFORMATION BOTTLENECK PRINCIPLE FOR SEQUENTIAL DATA.Here the aim is to show the relationship between FAVAE and the information bottleneck for sequential data.

Consider the information bottleneck object: DISPLAYFORM0 is expanded from Alemi et al. FORMULA0 to sequential data.

We need to distinguish betweenx 1:T and x 1:T , where x 1:T is the true distribution andx 1:T is the empirical distribution created by sampling from the true distribution.

We maximize the mutual information of true data x 1:T and z while constraining the information contained in the empirical data distribution.

We do this by using Lagrange multiplier: DISPLAYFORM1 where β is a constant.

For the first term, DISPLAYFORM2 where H (x 1:T ) is entropy, which can be neglected in optimization.

The last line is Monte Carlo approximation.

For the second term, DISPLAYFORM3 As a result, DISPLAYFORM4 For convenience of calculation, we use x i sampled from mini-batch data for both the reconstruction term and the regularization term.

This is only an approximation.

If the information bottleneck principle is followed completely, it is better to use different batch data for the reconstruction and regularization terms.

We expect the ladder network can disentangle representations at different levels of abstraction.

In this section, We check the factor extracted in each ladder by using 2D Reaching and 2D Wavy.

TAB1 shows the counting index of latent variable with the highest mutual information in each ladder network.

In TAB1 , the rows represent factor and the columns represent the index of the ladder networks.

The factor 1 (goal left / goal right) in 2D Reaching and the factor 1 (goal position) in 2D wavy Reaching were extracted to the most frequently in the latent variable in 3rd ladder.

Since the latent variables have 8 dimensions for the 1st ladder, 4 dimensions for the 2nd ladder and 2 dimensions for the 3rd ladder, the 3rd ladder should be the least frequent when factors are randomly entered for each z. Especially long-term and short-term factors are clear in the 2D wavy Reaching dataset.

In 2D Wavy Reaching dataset, there is distinct difference between factors of long and short time dependency.

The "goal position" is the factor which affect the entire trajectory, and other factors affect half length of the trajectory FIG7 ).In our experiment the goal of the trajectory which affect the entire trajectory tended to be expressed in the 3rd ladder.

In both datasets, only factor 1 represents goal positions while others represent shape of the trajectories.

Since factor 1 has different abstraction level from others, factor 1 and others result in different ladders such as ladder 3 and others.

C COMPARING WITH FHVAE FHVAE model is the recently proposed disentangled representation learning model.

We note that FHVAE model uses label information to disentangle time series data, which is different setup with our FAVAE model.

TAB2 shows a comparison of MIG and reconstruction using FHVAE as the baseline.

It was not possible to disentangle in 2D Reaching and 2D wavy using FHVAE, because LSTM used at the FHVAE can not learn data with very long sequences (sequence length of 1000).

For fair comparison with the FHVAE, we experimented with 2D Reaching (sequence length 100), 2D wavy Reaching (sequence length 100) at TAB2 .

In 2D Reaching, FHVAE model has the best score, while in 2D wavy Reaching FAVAE model with ladders and C has the best score.

Reaching.

The best C was decided by the value of KL divergence loss when we experimented to allow reconstruction with C = 0, 0, 0.

To evaluate the potential application of our model to robotic tasks, we constructed a robot endeffector simulation environment based on Bullet Real-Time Physics Simulation 2 .

The environment consisted of an end-effector, two balls (red and blue), and two baskets (red and blue) in a bi-dimensional space.

The end-effector grabs one of the balls and places it into the basket with the same color as the ball.

The data factors include movement "habits."

For example, the end-effector could reach the targeted ball by either directly moving toward the ball obliquely or first moving above of the ball and then lowering itself until it reached the ball (perpendicular movement).

The end-effector could drop the ball from far above the basket or place it gently in the basket.

Each factor could affect different lengths among the data; e.g., "the plan to place the ball in the basket" factor affects different lengths per datum since the initial distance from the target ball to the corresponding basket may differ.

This means that the time required to reach the ball should differ.

Note that input is a value such as gripper's position, not an image.

See Appendix B for a more detailed explanation.

FAVAE learned the disentangled factors of the Gripper dataset.

Example visualized latent traversals are shown in FIG5 .

The traversed latent variable determined which factors were disentangled, such as the initial position of blue ball and basket (b), the targeted ball (red or blue) (c), plan to reach the ball (move obliquely or move perpendicularly) (d), and plan to place the ball in the basket (drop the ball or placing it in the basket gently) (e).

These results indicate that our model can learn generative factors such as disentangled latent variables for robotic tasks, even though the length of the data affected by each factor may differ for each datum.

We used FAVAE in a two-ladder network with 12 and 8 latent variables with β=1000.

Sprits dataset is video data of video game "sprites".

It used was used in BID26 for confirming the extraction of disentangled representation between static factors and dynamic factors.

The dataset consists of sequences with T = 8 frames of dimension 3 × 64 × 64.

We use factors and motion of Sprites is shown in TAB3 and FIG8 .

We implemented the end-effector only rather than the entire robot arm since controlling the robot arm during the picking task is easily computable by calculating the inverse kinematics and inverse dynamics.

Gripper is a 12 dimensional data set: [joint x position, joint y position, finger1 joint position(angle), finger2 joint position(angle), box1 x position, box1 y position, box2 x position, box2 y position, ball1 x position, ball1 y position, ball2 x position, ball2 y position].

Eight factors are represented in this dataset: 1) color of ball to pick up, 2) initial location of red ball, 3) initial location of blue ball, 4) initial location of blue basket, 5) initial location of red basket, 6) plan for using end effector to move to ball to pick it up [first, moving horizontally to the x-location of ball and then descending horizontally to the y-location of ball, like the movement of the doll drawing machine (perpendicular motion); second, moving straight to the location of he ball to pick it up

<|TLDR|>

@highlight

We propose new model that can disentangle multiple dynamic factors in sequential data