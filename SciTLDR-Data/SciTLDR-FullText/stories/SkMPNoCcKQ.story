This work studies the problem of modeling non-linear visual processes by leveraging deep generative architectures for learning linear, Gaussian models of observed sequences.

We propose a joint learning framework, combining a multivariate autoregressive model and deep convolutional generative networks.

After justification of theoretical assumptions of inearization, we propose an architecture that allows Variational Autoencoders and Generative Adversarial Networks to simultaneously learn the non-linear observation as well as the linear state-transition model from a sequence of observed frames.

Finally, we demonstrate our approach on conceptual toy examples and dynamic textures.

While classification of image and video with Convolutional Neural Networks (CNN) is becoming an established practice, unsupervised learning and generative modeling remain to be challenging problems in deep learning.

A generative model of a visual process enables the possibility of generating sequences of video frames such that the appearance as well as the dynamics approximately resemble the original training process without copying it.

This procedure is typically referred to as video generation BID25 BID6 ) or video synthesis BID19 ).

More technically, this means that in addition to a suitable probability model for the individual frames, a probabilistic description for the frame-to-frame transition is also necessary.

Analysis and reproduction of visual processes simplifies considerably, if this transition can be described as a multivariate autoregressive (MAR) model, i.e., as a combination of linear transformations and Gaussian noise.

For instance, linear transformations are easily invertible and by means of spectral analysis, it can be studied how such a process behaves in the long term.

Realistically, most frame transitions in real-world visual processes unlikely are linear functions.

Nevertheless, unsupervised learning has come up with many approaches to fit MAR models to realworld processes, for instance by using linear low-rank approximations, as proposed by BID8 , or sparse approximations of the frames, as proposed by BID28 , or applying the kernel trick to them BID3 ).The success of Generative Adversarial Networks (GAN) introduced by BID9 and Variational Autoencoders (VAE) introduced by BID15 has led to an increased interest in deep generative learning and it seems natural to apply such techniques to sequential processes.

We approach this idea from the perspective of linearization, in order to keep the model as simple as possible.

In an analogous way as physicists transforming non-linear differential equations into linear ones by means of an appropriate change of variables, our approach is to learn latent representations of visual processes, such that the latent state-to-state transition can be described by an MAR model.

To this end, we jointly learn a non-linear observation and a linear state transition function by introducing a dynamic layer that can be used in conjunction with deep generative architectures such as GANs and VAEs.

In this paper, letters that appear both in roman and italicized type refer to random variables and realizations thereof, respectively.

If not stated otherwise, expected values of more than one random variables assume statistical independence.

The deep predictive coding network BID2 ), can be considered one of the earliest approaches of combining MAR models with deep learning.

More recently, such a combination was studied by BID18 .

The work by BID11 deals with linearizing transformations under uncertainty via neural networks.

It resembles our work in that we also focus on representation learning rather than on a particular type of process.

However, these works do not employ VAEs or GANs.

By contrast, BID27 combine Linear Dynamic Systems (LDSs) with VAEs.

However, the focus of their work is on control rather than on synthesis.

Furthermore, their model is locally linear and the transition distribution is modeled in the variational bound, whereas we model it as a separate layer.

This also is the main difference to the work of BID16 , in which VAEs are used as Kalman Filters.

The work of BID14 combines VAEs with linear dynamic models for forecasting images in video sequences.

Mathematically, it is a well-thought out approach, but proposes a training objective that is considerably more complex than the one proposed in this work.

GANs in combination with MAR models have been studied in experiments of BID12 , but the MAR model was learned separately from the GAN.Theoretical groundwork regarding learned visual transformations has been done by BID5 BID7 BID13 BID21 .

In particular, the dynamic layer proposed in this work bears resemblance to techniques employed in image-to-image translation BID17 ; ).The synthesis of video dynamics by means of neural networks has been discussed, among others by BID26 ; BID30 ; BID24 and BID29 .

We would like to emphasize the difference between video synthesis which is discussed in our work and the prediction of video frames as studied, for instance, by .

While the former refers to the problem of finding a probabilistic model for the spatial and temporal behavior of a visual process, the latter refers to finding a deterministic mapping from a set of previous frames to one or several future frames to come.

As a consequence, a video synthesis model needs to take care of long-term behavior.

Additionally, the probabilistic nature of video synthesis makes it considerably harder to evaluate the generated frames, since a unique ground truth cannot be provided, ruling out classical quantitative quality measures such as mean squared error or structured similarity.

Finally, the core contribution of this work is a combination of neural networks with Markov processes.

This has been the subject of many works in the recent past.

For a broad overview of results in this field, the reader is referred to Chapter 20 of the book Deep Learning by BID10 3 VISUAL PROCESSES

The Dynamic Texture (DT) model by BID8 has popularized LDSs in the modeling of visual processes.

Typically, an LDS is of the following form DISPLAYFORM0 where h t ??? R n is the low dimensional state space variable at time t, A ??? R n??n the state transition matrix, y t ??? R d the observation at time t and C ??? R d??n the observation matrix.

The vector y ??? R d represents a constant offset in the observation space.

The input term v t is modeled as i.i.d.

zero-mean Gaussian noise, and is independent of h t .Real-world visual processes are often highly non-linear and non-Gaussian, hence are considerably harder to work with.

In the following, we define a more general dynamic model that is applicable to a broad class of visual processes.

To keep the problem tractable, we make the assumption of two properties that were already implicitly assumed in the classical DT model.

The stationarity property guarantees that synthesis of new sequences will not diverge and the Markov property facilitates the statistical inference.

However, we believe that the method presented in this work can be generalized to Markov processes of higher orders.

We refer to Appendix A.2 for details.

(2) Again, y t ??? R d denotes the observation at time t. The self map ?? : M ??? M models the predicted frame transition and describes the deterministic part of the dynamics.

In this work, it is assumed to be differentiable.

Furthermore, the function ?? vt : M ??? M describes a displacement by v t in the in the tangent space T yt M of M at y t , followed by a retraction onto M. The displacement is sampled from i.i.d.

zero-mean Gaussian noise.

Eq. FORMULA0 is a special case of Eq. (2) , where ??(y) = CAC + (y ?????) +?? and ?? v (y) = y + Cv.

The model Eq. (2) describes a much broader class of visual processes than Eq. (1).

However, unlike model Eq. (2), the state transition in the linear model Eq.(1) enables straightforward prediction, generation, and analysis of observations.

It is thus of great interest to find a model that linearizes real-world visual processes, so that in some latent state space representation, the state transition admits the MAR model of the first line of Eq. (1).

Specifically, this work focuses on the following non-linear dynamic system model, i.e., a linear state transition and a non-linear observation mapping DISPLAYFORM0 where n < d and C : DISPLAYFORM1 Transforming a non-linear model to this form makes video synthesis as easy as sampling from autoregressive noise.

It is worth noticing that the model in Eq. FORMULA2 is not unique with respect to changes of basis in the state space BID8 ; BID0 ).

However, if C is implemented via a neural network, we can ensure that it accounts for a possible change of basis, e.g. by adding a linear layer to the input of the network.

We can thus make the following assumption on the latent samples h t without loss of generality.

Assumption 2.

The latent samples h t abide a standard normal distribution, i.e., h t ??? N ( ?? ; 0, I).

Remark 1.

If the state transition matrix A is given, and the process is stationary, i.e., p(h t ) = p(h t+1 ) for all t, then Assumption 2 essentially identifies the process noise model.

Namely, we have DISPLAYFORM2

In order to make sure that the latent states h t remain standard Gaussian in sequential synthesis scenarios, i.e., E ht+1 h t+1 h t+1 = I, we just need to ensure that the process noise is zero-mean and has the covariance matrix I ??? AA .

??? Before we propose an algorithm to jointly learn C and A by means of deep generative architecture, we further characterize the problem in order to investigate its feasibility and motivate our approach.

For a model Eq. (3) to substitute Eq. (2), it needs to transform the transition ?? to a multiplication by a matrix and the perturbation ?? vt to a superposition by zero-mean Gaussian noise.

The latter is generally easy to achieve, if the perturbation is sufficiently small and C is a diffeomorphism.

In the remainder of the section, we thus focus on linearization of ??.

A common linearization method in control theory is to approximate the dynamic system equation by a first-order Taylor polynomial around an equilibrium point BID22 ).

Clearly, for this to be possible, such a point needs to exist.

We thus propose the following assumption on the transition function ?? of Eq. (2).Assumption 3.

The differentiable self map ?? : M ??? M has at least one fixed point y * .However, such an approach might fail for real-world processes due to the curse of dimensionality.

A remedy is to employ an representation function ?? that maps the system observations to a lowerdimensional latent space prior to performing the linearization.

We call such a function a local linearizer.

DISPLAYFORM0 The map ?? is said to be a local linearizer at y * ??? M of ?? , if there exists a matrix ?? ??? R n??n such that the following equality holds true with y ??? R d .

DISPLAYFORM1 Generally speaking, local linearization is made possible by moving the fixed point to the origin of a new coordinate system.

More precisely, the following proposition holds.

DISPLAYFORM2 locally linearizes ??, if the rows and columns of the Jacobi matrix J ?? of ?? at y * lie in the row space of the Jacobi matrix J ?? of ?? at y * .Proof.

See Appendix A.1The Jacobi matrix property is only needed for consistency with Definition 1, where the limit y ??? y * is approached from any possible direction in R d .

It can be ignored, if we restrict the analysis entirely to the manifold M. Proposition 1 states that if ?? is a differentiable self map on M with a fixed point y * ??? M, and a diffeomorphism from M to R n exists, then there is a representation in which ?? can be approximated by a linear function for points on M that are not too far away from y * .Note that even if the Jacobi matrix J ?? can be sufficiently well estimated from the data, the linear approximation directly in the observation space R d will likely be worse than via reparametrization of a local linearizer ?? that functions as a chart of M. The reason is that predictions directly via J ?? can not be assumed to remain on M. Figuratively speaking, the aim of the local linearizer is to bend the space spanned by the rows and columns of J ?? to match the shape of M.

Local linearizability is a useful concept to analyze the feasibility of the problem at hand.

However, it does not take into account that two local linearizers can have significantly different linearization properties on a global scale.

Moreover, it does not provide any instructions on how to find an appropriate representation ?? and matrix ??. Traditionally, this task is approached by learning ?? and ?? separately while considering the sampled observations of the process globally, as will be described in the following.

To characterize the problem of global linearization, we need to introduce a measure.

It is sensible to consider the expected squared distance between the result of a transformation ?? and its linear approximation.

Since it is not possible to have an analytic expression for the distribution on the data manifold M, the latent space is often considered as a heuristic.

Because it is a common model assumption for deep generative models and in accordance with Assumption 2, we can assume standard Gaussian distribution for the latent space.

Let ?? : M ??? R n denote a data representation mapping and ?? : M ??? M the transition function to be linearized.

Consider the following expression DISPLAYFORM0 The denominator 2n is a normalization factor.

If ??(y) has standard Gaussian distribution, then q(??, ??, ??) < 1 denotes that the linear prediction with ?? is smaller than the expected distance of two independently drawn samples.

A common method of linearizing ??, is to first choose a representation ?? that is assumed to have good linearization properties, and then inferring ?? by minimizing Eq. (7) BID8 BID3 ; BID12 ).

In that case, the solution is given by?? DISPLAYFORM1 if ??(y) has standard Gaussian distribution.

However, the drawback of this approach is the difficulty to find an appropriate model for ?? that can be assumed to linearize the transformation ??. Moreover, separate learning implicitly assumes that a small linearization error in the embedded space will carry over to a small error in the observation space, but such an assumption is hard to justify given the high dimension of the problem.

We therefore propose to approach the problem by learning the representation and the linear transition jointly, by approximating the data distribution directly via deep generative models.

We now turn to the problem of finding a model Eq. (3) that approximates Eq. (2).

In the following, we will provide the preliminaries to lay out a training procedure to infer C and A from observed DISPLAYFORM0 In order to model Y by Eq. (3), we need to make sure that the probability distributions of y N and y N coincide for any N ??? N. By taking into account Assumption 1, this is equivalent to demanding that the joint probability of two succeeding frames coincide.

The joint probability distribution of h t and h t+1 is zero-mean Gaussian and, more specifically, DISPLAYFORM1 holds due to Assumption 2.

To summarize, we are looking for the function C and a matrix A, such that the random variable??? DISPLAYFORM2 has the same probability distribution as y 2 .

Estimation of probability distributions for high-dimensional data is still an ongoing research topic in deep learning.

The problem is typically framed as an approximation of a function f ?? , parameterized by a vector ?? in a finite-dimensional euclidean space.

The purpose of f ?? is to map realizations of low-dimensional, standard Gaussian noise to samples that abide the data distribution.

Typically, the function f ?? is realized by a neural network with trainable weights ??.

Most of the widely employed deep generative models, including the GAN, and the VAE, adopt this approach.

What these models differ in, is merely the way the network parameters ?? are trained.

For this work, we employ the Wasserstein GAN and the VAE to evaluate the proposed method.

Due to space constraints, we will not review these techniques here and refer the reader to BID1 for the Wasserstein GAN and to BID7 for the VAE.

For the following sections, it is sufficient to assume that these techniques are capable of approximating a function f ?? that maps from a low-dimensional standard Gaussian to a high-dimensional data distribution.

We now focus on how to learn a matrix A and a function C from a finite sequence y N ??? R d??N of a visual process such that it can be described by Eq. (3).

To this end, recall that, as discussed in Section 4.1, we only need to consider the joint probability of two succeeding observed frames.

The first step is thus to generate a training set of frame pairs from y N as {s 1 , . . .

, s N ???1 }, where DISPLAYFORM0 DISPLAYFORM1 (12) denotes a vectorized pair of frames.

We treat the samples as realizations of the random variable s and are looking for an architecture to learn the function C and a matrix A such that the probability distribution of the random variable DISPLAYFORM2 where h 1 , h 2 have the joint distribution described by Eq. FORMULA0 , coincides with the probability distribution of s, i.e. DISPLAYFORM3 It turns out that we can use a deep generative architecture to accomplish this task.

Remember, that a function f ?? learned by a deep generative model is capable of transforming samples x of standard Gaussian noise to samples from a high dimensional data distribution.

Let ?? = (A, B, ??) contain the matrices A, B ??? R n??n such that the constraint AA + BB = I n (15) is fulfilled, and the parameter ?? that defines the model for C ?? = C. Consider the following definition for f ?? .

DISPLAYFORM4 Assume that we can train f ?? , as defined in Eq. FORMULA0 , to map x ??? N (x; 0, I 2n ) tos = f ?? (x) with the probability distributions ??? p(s).(17) Then, A and C = C ?? indeed fulfill the condition in Eq. FORMULA0 for h 1 , h 2 with joint probability distribution Eq. (10).

In order to implement f ?? by a neural network, we propose the architecture depicted in FIG0 .

The first layer is linear and will be referred to as the dynamic layer in the following.

The dynamic layer implements the multiplication with the matrix DISPLAYFORM5 The output of the dynamic layer is split into an upper half h 1 and a lower half h 2 and both halves are fed to the subnetwork that implements the observer function C ?? .

The weights of the dynamic layer contain the matrices A, B. Thus, they can be trained along with ??, by back-propagation of the stochastic gradient.

However, we need to make sure that the stationarity constraint in Eq. FORMULA0 is not violated.

This can be done by adding a regularizer to the loss function L of the Deep Generative Network.

We thus substitute L(??) b??? DISPLAYFORM6 where ?? > 0 is a regularizer constant.

Here, L refers to the variational bound for the VAE or to the Generator loss for the (Wasserstein) GAN.

FORMULA0 ) as the foundation for implementing the observer function C ?? .

We use it in combination with an affine layer at the input.

This layer serves three purposes.

Firstly, it accounts for changes of bases in the latent space in order to conform with Assumption 2.

Secondly, it includes a bias so that the fixed point of the transition function can be transformed to the origin according to Proposition 1.

Finally, it further reduces the latent dimension to make the search for the transition matrix A feasible.

We train C ?? and A simultaneously by means of the dynamic layer introduced in Section 4.3.

To this end, we integrate the construction in FIG0 as the decoder of a VAE (without batch normalization) and the generator of a Wasserstein GAN (with batch normalization), respectively.

As the critic (discriminator) network of the Wasserstein GAN, we use the original discriminator of the DCGAN but double the number of output channels in order to match the dimensions of the training frame pairs.

As the encoder of the VAE, we use the discriminator of the DCGAN, but adapt the number of output channels to be 2n, where n is the latent dimension of the model.

The latent dimension is set to n = 10 for all experiments.

Synthesis is performed by sampling from the MAR model described by DISPLAYFORM7 and mapping the latent states to the observation space by means of C ?? .

The initial latent state h 0 is sampled from Gaussian white noise for the Wasserstein GAN experiments.

For the VAE experiments, it is estimated by applying the encoder to an observation frame pair from the training sequence.

PyTorch code for reproducing the experiments will be made available online upon publication, along with the entire set of results.

An isotropic standard Gaussian model N (s; f ?? (x), ?? 2 ) is assumed for the conditional data distribution.

The Adam optimizer with a step size of 2.5 * 10 ???4 is used to train the architecture.

The regularizer constant is set to ?? = 100.

We carry out two series of experiments.

In the first series of experiments, we use the MNIST dataset to generate repeating sequences of hand written digits.

We choose ?? = 4 and use sequences of length 10000 for training the network for 25 epochs.

Most of the number sequences we test can be well reproduced.

FIG3 depicts the synthesis results for the sequences 0123401234... and 4567845678....

Occasional jumps occur, due to the stochastic nature of the model.

An observation we make is that higher values of ?? improve the probability of synthesizing the correct sequence, but decrease the variability of digit shapes.

To investigate the linearization of geometrical transformations, we employ Category 0 of the Small NORB dataset that contains pictures of miniature animals under 6 different lighting conditions, 9 elevational and 18 azimuthal poses.

We train our architecture for 100 epochs with ?? = ??? 5 to synthesize sequences depicting azimuthal rotations of 20??? per frame.

Generally, the synthesis provides good results.

For each one of the five animals, the synthesis yields clearly recognizable rotation movements.

However, the algorithm tends to confuse pairs of opposite poses, as can be observed in Fig. 3a , where the poses are flipped after the fourth frame in each row.

With regard to Section 3.2, two explanations for this can be identified.

On the one hand, it is possible that the assumptions of Proposition 1 are not fulfilled, and no diffeomorphism, i.e. a bijective mapping exists from the image manifold to the embedded space.

In that case, the best we can do is finding a non-injective local diffeomorphism that assigns more than one point on the image manifold to a point in the embedded space.

On the other hand, it is thinkable, that an actual diffeomorphism exists, but that the euclidean distance implicitly minimized by the VAE is a bad estimation of the geodesic distance on the manifold.

Fig. 3b depicts the learned fixed point of the transition function.

It can be thought of as a rotationally invariant structure under limited exposure of light from above.

We employ the Wasserstein GAN for dynamic texture synthesis experiments and set the regularizer to ?? = 1.

We firstly test our method on the cropped UCLA-50 dataset of grayscale DTs.

The dataset contains 50 classes of 4 sequences of length 75 each.

We use one class at a time for the experiments.

The architecture is trained for 2000 epochs via RMSPROP with step size 2.5 * 10 ???4 .

Fig. 4 depicts the results for the classes candle and fountain-c-far.

Finally, we evaluate the method on three RGB sequences.

We run the optimization via RMSPROP with step size 5 * 10 ???5 for 900 epochs.

FIG5 depicts the result for the firepot, springwater and waterfountain sequences taken from BID29 .

Despite the low complexity of our model, we believe that the difference in quality compared to the state of the art results reported in BID29 is negligible.

This work presents an approach to learn embedded MAR models from image sequences.

We motivate the feasibility of this approach by introducing the concept of local linearizability and propose a joint learning procedure that employs deep generative models in combination with an additional linear component, the dynamic layer.

We report first positive results on low-resolution visual processes, where a first-order Markov property can be assumed, and hope to shed some light on the nature of linearization.

A possible future research direction is improving the theoretical understanding of linearizing representations and their applicability outside of stationary visual processes.

Let ?? ??? R n??n be a matrix.

Since ?? is a diffeomorphism, we define DISPLAYFORM0 holds.

Let us denote the Jacobian of ?? at y * by J ?? .

Because ?? maps y * to the origin, we can reformulate the requirement as DISPLAYFORM1 This requirement is fulfilled if the Jacobi matrices of ?? and ?? coincide, i.e. DISPLAYFORM2 The Jacobian of ?? at y * is given, according to he chain rule, by DISPLAYFORM3 where J ?? ???1 ??? R d??n is the Jacobian matrix of ?? ???1 at ??(y * ).

A matrix ?? ??? R n??n can be always found, such that Eq. FORMULA2 is fulfilled, if the columns and rows of J ?? lie in the column space of J ?? ???1 and the row space of J ?? , respectively.

The column space of J ?? ???1 coincides with the row space of J ?? , due to the identity J ?? J ?? ???1 = I n .

The statement of the proposition follows.

The proposed method is designed to linearize first-order Markov processes only.

This is in line with our empirical evaluation, in which the method was not capable of synthesizing sequences with persisting motions of non-constant velocities, e.g. the videos of running animals in BID29 .

However, the presented approach can be theoretically generalized to Markov processes of order m ??? 1.

We present a procedure that could be employed for this purpose.

Further investigation is necessary to evaluate the feasibility and practical applicability of such an approach.

The general procedure is as follows.1.

Set up an appropriate block matrix model for the joint probability of m + 1 succeeding observations.

2. Build a dynamic layer that performs a multiplication with a lower-triangular (m + 1) ?? (m + 1) block matrix F .3.

Introduce regularizers to preserve the block Toeplitz structure of the covariance matrix, 4.

Compute the MAR parameters from the resulting covariance matrix.

We illustrate the procedure for the case m = 2.1.

Due to the stationarity assumption, the covariance matrix for three succeeding observations h t , h t+1 , h t+2 has the form DISPLAYFORM0 2.

The matrix describing the dynamic layer has the form DISPLAYFORM1 The outputs of the dynamic layer will thus have the distribution DISPLAYFORM2 We thus need to achieve DISPLAYFORM3 3.

This can be ensured by using the regularizer DISPLAYFORM4 with ?? 1 , ?? 2 , ?? 3 > 0.

4.

A second-order MAR model has the form h t+2 = A 0 h t + A 1 h t+1 + Bv t , v t ??? N (v t , 0, I n ).By our assumptions on the process, this yields the system of equations, Thus, we can infer A 0 , A 1 and B via solving DISPLAYFORM5

@highlight

We model non-linear visual processes as autoregressive noise via generative deep learning.

@highlight

Proposes a new method that models non-linear visual process with a deep version of a linear process (Markov process).

@highlight

This paper proposes a new deep generative model for sequences, particularly image sequences and video, which uses a linear structure in part of the model.