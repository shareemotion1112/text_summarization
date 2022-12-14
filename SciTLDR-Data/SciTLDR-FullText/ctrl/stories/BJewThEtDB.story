Clustering algorithms have wide applications and play an important role in data analysis fields including time series data analysis.

The performance of a clustering algorithm depends on the features extracted from the data.

However, in time series analysis, there has been a problem that the conventional methods based on the signal shape are unstable for phase shift, amplitude and signal length variations.

In this paper, we propose a new clustering algorithm focused on the dynamical system aspect of the signal using recurrent neural network and variational Bayes method.

Our experiments show that our proposed algorithm has a robustness against above variations and boost the classification performance.

The rapid progress of IoT technology has brought huge data in wide fields such as traffic, industries, medical research and so on.

Most of these data are gathered continuously and accumulated as time series data, and the extraction of features from a time series have been studied intensively in recent years.

The difficulty of time series analysis is the variation of the signal in time which gives rise to phase shift, compress/stretch and length variation.

Many methods have been proposed to solve these problems.

Dynamic Time Warping (DTW) was designed to measure the distance between warping signals (Rabiner & Juang, 1993) .

This method solved the compress/stretch problem by applying a dynamic planning method.

Fourier transfer or wavelet transfer can extract the features based on the frequency components of signals.

The phase shift independent features are obtained by calculating the power spectrum of the transform result.

In recent years, the recurrent neural network (RNN), which has recursive neural network structure, has been widely used in time series analysis (Elman, 1990; 1991) .

This recursive network structure makes it possible to retain the past information of time series.

Furthermore, this architecture enables us to apply this algorithm to signals with different lengths.

Although the methods mentioned above are effective solutions for the compress/stretch, phase shift and signal length variation issues respectively, little has been studied about these problems comprehensively.

Let us turn our attention to feature extraction again.

Unsupervised learning using a neural network architecture autoencoder (AE) has been studied as a feature extraction method (Hinton & Salakhutdinov, 2006; Vincent et al., 2008; Rifai et al., 2011) .

AE using RNN structure (RNN-AE) has also been proposed (Srivastava et al., 2015) and it has been applied to real data such as driving data (Dong et al., 2017) and others.

RNN-AE can be also interpreted as the discrete dynamical system: chaotic behavior and the deterrent method have been studied from this point of view (Zerroug et al., 2013; Laurent & von Brecht, 2016) .

In this paper, we propose a new clustering algorithm for feature extraction focused on the dynamical system aspect of RNN-AE.

In order to achieve this, we employed a multi-decoder autoencoder with multiple decoders to describe different dynamical systems.

We also applied the variational Bayes method (Attias, 1999; Ghahramani & Beal, 2001; Kaji & Watanabe, 2011) as the clustering algorithm.

This paper is composed as follows: in Section 4, we explain AE from a dynamical system view, then we define our model and from this, derive its learning algorithm.

In Section 5, we describe the application of our algorithm to an actual time series to show its robustness, including running two experiments using periodic data and driving data.

Finally we summarize our study and describe our future work in Section 7.

A lot of excellent clustering/representation algorithms of data using AE have been studied so far (Tschannen et al., 2018) .

Song et al. (2013) integrated the distance between data and centroids into an objective function to obtain a cluster structure in the encoded data space.

Pineau & Lelarge (2018) proposed a generative model based on the variational autoencoder (VAE) (Kingma & Welling, 2014) with a clustering structure as a prior distribution.

Wang et al. (2019) achieved a high separability clustering result by adding a regularization term for the orthogonality and balanced clusters of the encoded data.

These, however, are regularization methods of the objective function, and focused on only the distribution of the encoded data.

They did not give the clustering policy based on the decoder structure, namely, the reconstruction process of the data.

From dynamical system point of view, one decoder of RNN-AE corresponds to a single dynamics in the space of latent representation.

Hence, it is natural to equip RNN-AE with multiple decoders to implement multiple dynamics.

Such an extension of RNN-AE, however, has yet to be proposed in related works to the best of our knowledge.

3 RECURRENT NEURAL NETWORK AND DYNAMICAL SYSTEM 3.1 RECURRENT NEURAL NETWORK USING UNITARY MATRIX RNN is a neural network designed for time series data.

The architecture of the main unit is called cell, and mathematical expressions are shown in Fig. 1 and Eq. (1).

Suppose we are given a time series,

where D denotes data dimension.

RNN, unlike the usual feed-forward neural network, operates the same transform matrix to the hidden valuable recursively,

where ??(??) is an activation function and z t , h t , b ??? R L .

This recursive architecture makes it possible to handle signals with different lengths, although it is vulnerable to the vanishing gradient problem as with the deep neural network (DNN) (Elman, 1990; 1991) .

Long short-term memory (LSTM) and gated recurrent unit (GRU) are widely known solutions to this problem (Gers et al., 2000; Hochreiter & Schmidhuber, 1997; Cho et al., 2014) .

These methods have the extra mechanism called a gate structure to control output scaling and retaining/forgetting of the signal information.

Though this mechanism works effectively in many application fields (Malhotra et al., 2015; Rana, 2016) , the architecture of network is relatively complicated.

As an alternative simpler method to solve this problem, the algorithm using a unitary matrix as the transfer matrix V was proposed in recent years.

Since the unitary matrix does not change the norm of the variable vector, we avoid the vanishing gradient problem.

In addition, the network architecture remains unchanged from the original RNN.

In this paper, we focus on the dynamical system aspect of the original RNN.

We employ the unitary matrix type RNN to take advantage of this dynamical system structure.

However, to implement the above method, we need to find the transform matrix V in the space of unitary matrices

is the set of complex-valued general linear matrices with size L ?? L and * means the adjoint matrix.

Several methods to find the transform matrix from the U has been reported so far (Pascanu et al., 2013; Jing et al., 2017; Wisdom et al., 2016; Arjovsky et al., 2016; Jing et al., 2019) .

Here, we adopt the method proposed by Jing et al. (2017) .

The architecture of AE using RNN is shown in Fig. 2 .

AE is composed of an encoder unit and a decoder unit.

The parameters

where X is the input data and X dec is the decoded data.

The input data is recovered from only the encoded signal h using the matrix (V dec , U dec ), therefore h is considered as the essential information of the input signal.

When focusing on the transformation Figure 2 : Architecture of RNN Autoencoder of the hidden variable, this recursive operation has the same structure of a discrete dynamical system expression as described in the following equation:

where f is given by Eq. (1).

From this point of view, we can understand that RNN describes the universal dynamical system structure which is common to the all input signals.

In this section, we will give the architecture of the Multi-Decoder RNN AE (MDRA) and its learning algorithm.

As we discussed in the previous section, RNN can extract the dynamical system characteristics of the time series.

In the case of the original RNN, the model expresses just one dynamical system, hence all input data are recovered from the encoded result h by the same recovery rule.

Therefore h is usually used as the feature value of the input data.

In contrast, in this paper, we focus on the transformation rule itself.

For this purpose, we propose MDRA which has multiple decoders to extract various dynamical system features.

The architecture of MDRA is shown in Fig. 3 .

Let us put

We will derive the learning algorithm to optimize the whole set of parameters W in the following section.

We applied a clustering method to derive the learning algorithm of MDRA.

Many clustering algorithms have been proposed: here we employ the variational Bayes (VB) method, because the VB method enabled us to adjust the number of clusters by tuning the hyperparameters of a prior distribution.

We first define free energy, which is negative marginal log-likelihood, by the following equation,

where X is data tensor defined in Section 3 and W is parameter tensor of MDRA defined above.

is the set of latent variables each of which means an allocation for a decoder.

That is, y n = (y n1 , ?? ?? ?? , y nK ) T ??? R K , where y nk = 1 if X n is allocated to the k-th decoder

is the probability density function representation of MDRA parametrized by tensor W, p(??) and p(??) are its prior distributions for a probability vector ?? = (?? 1 , ?? ?? ?? , ?? K ) and a precision parameter ?? > 0.

We applied the Gaussian mixture model as our probabilistic model.

Hence p(??) and p(??) were given by Dirichlet and gamma distributions respectively which are the conjugate prior distributions of multinomial and Gaussian distributions.

These specific distributions are given as follows:

Here, ?? 0 > 0, ?? 0 > 0 and ?? 0 > 0 are hyperparameters and g(h n |W k dec ) = X n dec,k denotes decoder mapping of RNN from the encoded n-th data h n , H = (h 1 , ?? ?? ?? , h N ) and T n D is the total signal dimension of input signal X n including dimension of input data.

To apply the variational Bayes algorithm, we the derive the upper bound of the free energy by applying Jensen's inequality,

where D KL (???????) is the Kullback???Leibler divergence.

The upper boundF (X|W) is called the variational free energy or (negated) evidence lower bound (ELBO).

The variational free energy is minimized using the variational Bayes method under the fixed parameters W. Furthermore, it is also minimized with respect to the parameters W by applying the RNN learning algorithm to the second term ofF (X|W),

In this section, we derive the variational Bayes algorithm for MDRA to minimize the variational free energy.

We show the outline of the derivation below (for a detailed derivation, see Appendix A).

The general formula of the variational Bayes algorithm is given by

.

By applying the above equations to the above probabilistic models, we obtained the specific algorithm shown in Appendix A.1.

Then we minimize the following term using RNN algorithm:

From the above discussion, we finally obtained the following MDRA algorithm.

Fig. 4

Calculate R = (r nk ) by the algorithm VB part of MDRA (Algorithm 2).

until the difference of variational free energyF (X|W) < Threshold We first examined the basic performance of our algorithm using periodic signals.

Periodic signals are typical time series signals expressed by dynamical systems.

Input signals have 2, 4, and 8 periods respectively in 64 steps.

Each signal is added a phase shift (maximum one period), amplitude variation (from 50% to 100% of the maximum amplitude), additional noise (maximum 2% of maximum amplitude) and signal length variation (maximum 80% of the maximum signal length).

Examples of input data are illustrated in Fig. 5 .

We compared RNN-AE to MDRA on its feature extraction performance using the above periodic signals.

Fig. 6 and Fig. 7 show the results of RNN-AE and MDRA respectively.

The parameter Table 3 in Appendix B. We used multi-dimensional scaling (MDS) as the dimension reduction method to visualize the distributions of features in Fig. 6 and Fig. 7.

Fig. 6 shows the distribution of the encoded data h n which is the initial value of the decoder unit in Fig. 2 .

We found that RNN-AE can separate the input data into three regions corresponding to each frequency.

However each frequency region is distributed widely, therefore some part of the region overlap each other.

Fig.7 shows the distributions of the encoded data h n and the clustering allocation weight r n extracted by MDRA.

The distribution of r n , shown in the right figure of Fig. 7 , is completely separated into each frequency component without overlap.

This result shows that the distribution of r n as the feature extraction has robustness for a phase shift, amplitude and signal length variation.

We also show that MDRA can boost the classification accuracy using an actual driving data in the next section.

We applied our algorithm to a driver identification problem.

We use the data consisting of 3 drivers signals when driving, including speed, acceleration, braking and steering angle signals.

1 The input signal was about 10 seconds differential data (128 steps), which was cut out from the original data by a sliding window.

2 The detailed information of the input data is shown in Table 1 .

We also show samples of data (difference of acceleration) in Fig. 10 in Appendix C. The feature extraction results by RNN-AE and MDRA are shown in Fig. 8 .

The parameter setting of this experiment is listed in Table 4 2 We use only the data of which the maximum acceleration difference is more than a certain threshold.

and MDRA respectively, the right figure is the distribution of r n of MDRA.

We can find different trends in the distributions of the latent variable h n and r n of MDRA.

The distribution of r n spreads wider than that of h n .

Table 2 shows the accuracy of the driver identification results using the above

We verified the feature extraction peformance of the MDRA using actual time series data.

In Section 5.1, we saw that the periodic signals are completely classified by the frequency using clustering weight r n .

In this experiment, the average clustering weights, the elements of are (3.31e-01, 8.31e-47, 8.31e-47, 3.46e-01, 8.31e-47, 3.19e-01, 8.31e-47) , with only three components having effective weights.

This weight narrowing-down is one of the advantages of VB learning.

The left of Fig. 9 shows an enlarged view of around "freq 4" in Fig. 7 (right) .

We found that the distribution of "freq 4" is in fact spread linearly.

The right of Fig. 9 is the result Hinton, 2008) .

We found that each frequency data formed several spreading clusters without overlapping.

As we saw earlier, the distribution of r n has a spreading distribution compared to that of h n .

We inferred that the spreading of the distribution r n was caused by extracting the diversity on the driving scenes.

In addition, the identification result shows that the combination of the features given by r n and h n can improve the performance.

Dong et al. (2017) , which studied a driver identify algorithm using the AE, proposed the minimization of the error integrating the reconstruction error of AE and the classification error of deep neural network.

This algorithm can avoid the over-fitting by using unlabeled data whose data collection cost is smaller than labeled data.

From these results, we can expect that the MDRA can contribute to not only boost identification performance but also restrain the over-fitting.

In this paper, we proposed a new algorithm MDRA that can extract dynamical system features of a time series data.

We conducted experiments using periodic signals and actual driving data to verify the advantages of MDRA.

The results show that our algorithm not only has robustness for the phase shift, amplitude and signal length variation, but also can boost classification performance.

The phase transition phenomenon of the variational Bayes learning method, depending on the hyperparameters, has been reported in (Watanabe & Watanabe, 2006) The hyperparameter setting of the prior distribution has a great effect on the clustering result and classification performance.

We intend to undertake a detailed study of the relation between the feature extraction performance and hyperparameter setting of the prior distributions in the future.

In addition,

where T n D means total signal dimension.

Therefore, we obtain

We here put

Hence

Next we calculate log q(??, ??),

Above equation can be divided into the two terms including ?? and ?? respectively, E q(yn) [y nk ] = 1 ?? q(y nk = 1) + 0 ?? q(y nk = 0) = q(y nk = 1) = r nk to the above equation, we obtain

On the other hand,

Similarly, by applying E q(yn) [y nk ] = r nk , we obtain log q(??)

We finally calculate log ?? nk in Eq. (3).We first calculate log q(??) and log q(??), Gamma(??|??,??) .

By using the expectations of ?? and log ?? by gamma distribution

Similarly, q(??) turns out to be the Dirichlet distribution with parameters (?? 1 , ?? ?? ?? ,?? K ), and

is calculated by the same way in the general mixture model.

Therefore we finally obtain

From the above results, the following variational Bayes algorithm is derived.

A

where we used r nk = E q(yn) [y nk ].

We achieve this by applying RNN algorithm.

From the above discussion including Appendix A.1, we obtain the MDRA algorithm.

By putting ?? = e x , we obtain x = log ??, d?? = e x dx,

In this section, we show the parameter setting of the experiments in Section 5.

Here L is the dimension of hidden variable h, capacity, fft and cpx are parameters of EUNN (Jing et al., 2017) , K is the number of the decoders, ?? 0 , ?? 0 , ?? 0 are hyperparameters of prior distributions.

We applied our algorithm to the driving data in the Section 5.2.

We used the differential signals of speed, acceleration, braking and steering angle signals in the experiment.

Fig. 10 shows examples of acceleration signals.

<|TLDR|>

@highlight

Novel time series data clustring algorithm based on dynamical system features.