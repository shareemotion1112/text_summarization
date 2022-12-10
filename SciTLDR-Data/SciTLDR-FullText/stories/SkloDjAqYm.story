Neuronal assemblies, loosely defined as subsets of neurons with reoccurring spatio-temporally coordinated activation patterns, or "motifs", are thought to be building blocks of neural representations and information processing.

We here propose LeMoNADe, a new exploratory data analysis method that facilitates hunting for motifs in calcium imaging videos, the dominant microscopic functional imaging modality in neurophysiology.

Our nonparametric method extracts motifs directly from videos, bypassing the difficult intermediate step of spike extraction.

Our technique augments variational autoencoders with a discrete stochastic node, and we show in detail how a differentiable reparametrization and relaxation can be used.

An evaluation on simulated data, with available ground truth, reveals excellent quantitative performance.

In real video data acquired from brain slices, with no ground truth available, LeMoNADe uncovers nontrivial candidate motifs that can help generate hypotheses for more focused biological investigations.

Seventy years after being postulated by Hebb (1949) , the existence and importance of reoccurring spatio-temporally coordinated neuronal activation patterns (motifs), also known as neuronal assemblies, is still fiercely debated BID12 Singer, 1993; BID17 Ikegaya et al., 2004; Cossart & Sansonetti, 2004; BID4 BID13 BID20 Stevenson & Kording, 2011; BID0 Carrillo-Reid et al., 2015) .

Calcium imaging, a microscopic video technique that enables the concurrent observation of hundreds of neurons in vitro and in vivo (Denk et al., 1990; Helmchen & Denk, 2005; Flusberg et al., 2008) , is best suited to witness such motifs if they indeed exist.

Cell Identification e.g. BID25 ; Zhou et al. (2018) Spike Time Extraction e.g. Speiser et al. (2017) Motif Detection e.g. BID26 ; BID21 Neuronal Assemblies Motif Detection in Ca Videos LeMoNADe FIG4 : We present LeMoNADe, a novel approach to identify neuronal assemblies directly from calcium imaging data.

In contrast to previous methods, LeMoNADe does not need pre-processing steps such as cell identification and spike time extraction for unravelling assemblies.

In recent years, a variety of methods have been developed to identify neuronal assemblies.

These methods range from approaches for the detection of synchronous spiking, up to more advanced methods for the detection of arbitrary spatio-temporal firing patterns (Comon, 1994; BID16 Grün et al., 2002a; BID0 BID8 BID26 BID21 .

All of these methods, however, require a spike time matrix as input.

Generating such a spike time matrix from calcium imaging data requires the extraction of individual cells and discrete spike times.

Again, many methods have been proposed for these tasks BID14 Diego et al., 2013; Diego & Hamprecht, 2013; BID18 BID24 Diego & Hamprecht, 2014; Kaifosh et al., 2014; BID25 BID1 Inan et al., 2017; Spaen et al., 2017; BID6 Speiser et al., 2017; Zhou et al., 2018) .

Given the low signal-to-noise ratios (SNR), large background fluctuations, non-linearities, and strong temporal smoothing due to the calcium dynamics itself as well as that of calcium indicators, it is impressive how well some of these methods perform, thanks to modern recording technologies and state-of-the-art regularization and inference BID25 Zhou et al., 2018) .

Still, given the difficulty of this data, errors in segmentation and spike extraction are unavoidable, and adversely affect downstream processing steps that do not have access to the raw data.

Hence, properly annotating data and correcting the output from automatic segmentation can still take up a huge amount of time.

In this paper, we propose LeMoNADe (Learned Motif and Neuronal Assembly Detection), a variational autoencoder (VAE) based framework specifically designed to identify repeating firing motifs with arbitrary temporal structure directly in calcium imaging data (see FIG4 ).

The encoding and decoding networks are set up such that motifs can be extracted directly from the decoding filters, and their activation times from the latent space (see sec. 3).

Motivated by the sparse nature of neuronal activity we replace the Gaussian priors used in standard VAE.

Instead we place Bernoulli priors on the latent variables to yield sparse and sharply peaked motif activations (sec. 3.1).

The choice of discrete Bernoulli distributions makes it necessary to use a BinConcrete relaxation and the Gumbel-softmax reparametrization trick BID11 Jang et al., 2017) to enable gradient descent techniques with low variance (sec. 3.3).

We add a β-coefficient (Higgins et al., 2017) to the loss function in order to adapt the regularization to the properties of the data (sec. 3.3).

Furthermore, we propose a training scheme which allows us to process videos of arbitrary length in a computationally efficient way (sec. 3.4).

On synthetically generated datasets the proposed method performs as well as a state-of-the-art motif detection method that requires the extraction of individual cells (sec. 4.1).

Finally, we detect possible repeating motifs in two fluorescent microscopy datasets from hippocampal slice cultures (sec. 4.2).

A PyTorch implementation of the proposed method is released at https://github.com/EKirschbaum/LeMoNADe.

Autoencoder and variational autoencoder Variational Autoencoders (VAEs) were introduced by Kingma & Welling (2014) and have become a popular method for unsupervised generative deep learning.

They consist of an encoder, mapping a data point into a latent representation, and a decoder whose task is to restore the original data and to generate samples from this latent space.

However, the original VAE lacks an interpretable latent space.

Recent suggestions on solving this problem have been modifications of the loss term (Higgins et al., 2017) or a more structured latent space (Johnson et al., 2016; Deng et al., 2017) .VAE have also been successfully used on video sequences.

BID7 learn a disentangled representation to manipulate content in cartoon video clips, while Goyal et al. (2017) combine VAEs with nested Chinese Restaurant Processes to learn a hierarchical representation of video data.

Johnson et al. (2016) use a latent switching linear dynamical system (SLDS) model combined with a structured variational autoencoder to segment and categorize mouse behavior from raw depth videos.

Unfortunately, this model is not directly applicable to the task of identifying motifs with temporal structure from calcium imaging data for the following reasons: Firstly, neuronal assemblies are expected to extend over multiple frames.

Since in the model by Johnson et al. (2016) the underlying latent process is a relatively simple first-order Markovian (switching) linear process, representing longer-term temporal dependencies will be very hard to achieve due to the usually exponential forgetting in such systems.

Secondly, in the model of Johnson et al. (2016) each frame is generated from exactly one of M latent states.

For calcium imaging, however, most frames are not generated by one of the M motifs but from noise, and different motifs could also temporally overlap which is also not possible in the model by Johnson et al. (2016) .Closest to our goal of detecting motifs in video data is the work described in BID2 .

In this approach, a convolutional autoencoder is combined with a number of functions and regularization terms to enforce interpretability both in the convolutional filters and the latent space.

This method was successfully used to detect patterns in data with document structure, including optical flow features of videos.

However, as the cells observed in calcium imaging are spatially stationary and have varying luminosity, the extraction of optical flow features makes no sense.

Hence this method is not applicable to the task of detecting neuronal assemblies in calcium imaging data.

Cell segmentation and spike time extraction from calcium imaging data Various methods have been proposed for automated segmentation and signal extraction from calcium imaging data.

Most of them are based on non-negative matrix factorization BID14 Diego & Hamprecht, 2014; BID25 Inan et al., 2017; Zhou et al., 2018 ), clustering (Kaifosh et al., 2014 Spaen et al., 2017) , and dictionary learning (Diego et al., 2013; Diego & Hamprecht, 2013; BID18 .

Recent approaches started to use deep learning for the analysis of calcium imaging data.

BID1 and BID6 use convolutional neural networks (CNNs) to identify neuron locations and Speiser et al. (2017) use a VAE combined with different models for calcium dynamics to extract spike times from the calcium transients.

Although many sophisticated methods have been proposed, the extraction of cells and spike times from calcium imaging data can still be prohibitively laborious and require manual annotation and correction, with the accuracy of these methods being limited by the quality of the calcium recordings.

Furthermore, some of the mentioned methods are specially designed for two-photon microscopy, whereas only few methods are capable to deal with the low SNR and large background fluctuations in single-photon and microendoscopic imaging (Flusberg et al., 2008; Ghosh et al., 2011) .

Additional challenges for these methods are factors such as non-Gaussian noise, non-cell background activity and seemingly overlapping cells which are out of focus (Inan et al., 2017) .Neuronal assembly detection The identification of neuronal assemblies in spike time matrices has been studied from different perspectives.

For the detection of joint (strictly synchronous) spike events across multiple neurons, rather simple methods based on PCA or ICA have been proposed (Comon, 1994; BID16 BID8 , as well as more sophisticated statistical methods such as unitary event analysis (Grün et al., 2002a; BID0 .

Higher-order correlations among neurons and sequential spiking motifs such as synfire chains can be identified using more advanced statistical tests (Staude et al., 2010a; BID0 Gerstein et al., 2012) .

The identification of cell assemblies with arbitrary spatio-temporal structure has been addressed only quite recently.

One approach recursively merges sets of units into larger groups based on their joint spike count probabilities evaluated across multiple different time lags BID26 .

Another method uses sparse convolutional coding (SCC) for reconstructing the spike matrix as a convolution of spatio-temporal motifs and their activations in time BID21 ).

An extension of this method uses a group sparsity regularization to identify the correct number of motifs BID10 .

DISPLAYFORM0 Figure 2: Schematic sketch of the proposed method.

In this toy example, the input video x is an additive mixture of two motifs (highlighted in red and blue) plus noise, as shown in (a).

To learn the motifs and activations, the loss between input video x and reconstructed video x is minimized.

(b) shows the generation of the reconstructed video through the proposed VAE framework.

To the authors' knowledge, solely Diego & Hamprecht (2013) address the detection of neuronal assemblies directly from calcium imaging data.

This method, however, only aims at identifying synchronously firing neurons, whereas the method proposed in this paper can identify also assemblies with more complex temporal firing patterns.

LeMoNADe is a VAE based latent variable method, specifically designed for the unsupervised detection of repeating motifs with temporal structure in video data.

The data x is reconstructed as a convolution of motifs and their activation time points as displayed in figure 2a .

The VAE is set up such that the latent variables z contain the activations of the motifs, while the decoder encapsulates the firing motifs of the cells as indicated in figure 2b .

The proposed generative model is displayed in figure 3 .

The great benefit of this generative model in combination with the proposed VAE is the possibility to directly extract the temporal motifs and their activations and at the same time take into account the sparse nature of neuronal assemblies.

In the proposed model the dataset consists of a single video x ∈ R T ×P ×P with T frames of P × P pixels each.

We assume this video to be an additive mixture of M repeating motifs of maximum temporal length F .

At each time frame t = 1, . . .

, T , and for each motif m = 1, . . .

, M , a latent random variable z m t ∈ {0, 1} is drawn from a prior distribution p a (z).

The variable z m t indicates

In order to learn the variational parameters, the KL-divergence between approximate and true posterior KL(q φ (z | x) p θ (z | x)) is minimized.

Instead of minimizing this KL-divergence, we can also maximize the variational lower bound L(θ, φ; x) (ELBO) (see e.g. BID3 ) DISPLAYFORM0 In order to optimize the ELBO, the gradients w.r.t.

the variational parameters φ and the generative parameters θ have to be computed.

The gradient w.r.t.

φ, however, cannot be computed easily, since the expectation in eq. (2) depends on φ.

A reparameterization trick BID5 is used to overcome this problem: the random variable z ∼ q φ (z | x) is reparameterized using a differentiable transformation h φ (ε, x) of a noise variable ε such that DISPLAYFORM1 The reparameterized ELBO, for which the expectation can be computed, e.g. using Monte Carlo sampling, is then given by DISPLAYFORM2 More details on VAE as introduced by Kingma & Welling (2014) are given in appendix A.

In our case, however, by sampling from Bernoulli distributions we have added discrete stochastic nodes to our computational graph, and we need to find differentiable reparameterizations of these nodes.

The Bernoulli distribution can be reparameterized using the Gumbel-max trick BID9 Yellott, 1977; BID19 Hazan & Jaakkola, 2012; BID11 .

This, however, is not differentiable.

For this reason we use the BinConcrete distribution BID11 , which is a continuous relaxation of the Bernoulli distribution with temperature parameter λ.

For λ → 0 the BinConcrete distribution smoothly anneals to the Bernoulli distribution.

The BinConcrete distribution can be reparameterized using the Gumbel-softmax trick BID11 Jang et al., 2017) , which is differentiable.

BID11 show that for a discrete random variable z ∼ Bernoulli(α), the reparameterization of the BinConcrete relaxation of this discrete distribution is DISPLAYFORM0 where DISPLAYFORM1 Hence the relaxed and reparameterized lower boundL(θ,α; x) ≈ L(θ, φ; x) can be written as DISPLAYFORM2 where gα ,λ1 (y | x) is the reparameterized BinConcrete relaxation of the variational posterior q φ (z | x) and fã ,λ2 (y) the reparameterized relaxation of the prior p a (z).

λ 1 and λ 2 are the respective temperatures andα andã the respective locations of the relaxed and reparameterized variational posterior and prior distribution.

The first term on the RHS of eq. FORMULA6 is a negative reconstruction error, showing the connection to traditional autoencoders, while the KL-divergence acts as a regularizer on the approximate posterior q φ (z | x).

As shown in Higgins et al. FORMULA1 , we can add a β-coefficient to this KL-term which allows to vary the strength of the constraint on the latent space.

Instead of maximizing the lower bound, we will minimize the corresponding loss function DISPLAYFORM3 with MSE(x, x ) being the mean-squared error between x and x , and the β-coefficient β KL .

Datasets with low SNR and large background fluctuations will need a stronger regularization on the activations and hence a larger β KL than higher quality recordings.

Hence, adding the β-coefficient to the loss function enables our method to adapt better to the properties of specific datasets and recording methods.

The encoder network starts with a few convolutional layers with small 2D filters operating on each frame of the video separately, inspired by the architecture used in BID1 to extract cells from calcium imaging data.

Afterwards the feature maps of the whole video are passed through a final convolutional layer with 3D filters.

These filters have the size of the feature maps obtained from the single images times a temporal component of length F , which is the expected maximum temporal length of the motifs.

We apply padding in the temporal domain to also capture motifs correctly which are cut off at the beginning or end of the analyzed image sequence.

The output of the encoder are the parametersα which we need for the reparametrization in eq. (5).

From the reparametrization we gain the activations z which are then passed to the decoder.

The decoder consists of a single deconvolution layer with M filters of the original frame size times the expected motif length F , enforcing the reconstructed data x to be an additive mixture of the decoder filters.

Hence, after minimizing the loss the filters of the decoder contain the detected motifs.

Performing these steps on the whole video would be computationally very costly.

For this reason, we perform each training epoch only on a small subset of the video.

The subset consists of a few hundred consecutive frames, where the starting point of this short sequence is randomly chosen in each epoch.

We found that doing so did not negatively affect the performance of the algorithm.

By using this strategy we are able to analyse videos of arbitrary length in a computationally efficient way.

More implementation details can be found in appendix B.

The existence of neuronal assemblies is still fiercely debated and their detection would only be possible with automated, specifically tailored tools, like the one proposed in this paper.

For this reason, no ground truth exists for the identification of spatio-temporal motifs in real neurophysiological spike data.

In order to yet report quantitative accuracies, we test the algorithm on synthetically generated datasets for which ground truth is available.

For the data generation we used a procedure analogous to the one used in Diego et al. FORMULA1 and Diego & Hamprecht (2013) for testing automated pipelines for the analysis and identification of neuronal activity from calcium imaging data.

In contrast to them, we include neuronal assemblies with temporal firing structure.

The cells within an assembly can have multiple spikes in a randomly chosen but fixed motif of temporal length up to 30 frames.

We used 3 different assemblies in each sequence.

Additionally, spurious spikes of single neurons were added to simulate noise.

The ratio of spurious spikes to all spikes in the dataset was varied from 0% up to 90% in ten steps.

The details of the synthetic data generation can be found in appendix C.1.To the best of our knowledge, the proposed method is the first ever to detect video motifs with temporal structure directly in calcium imaging data.

As a consequence, there are no existing baselines to compare to.

Hence we here propose and evaluate the SCC method presented in Peter et al. FORMULA1 as a baseline.

The SCC algorithm is able to identify motifs with temporal structure in spike trains or calcium transients.

To apply it to our datasets, we first have to extract the calcium transients of the individual cells.

For the synthetically generated data we know the location of each cell by construction, so this is possible with arbitrary accuracy.

The output of the SCC algorithm is a matrix that contains for each cell the firing behavior over time within the motif.

For a fair comparison we brought the motifs found with LeMoNADe, which are short video sequences, into the same format.

The performance of the algorithms is measured by computing the cosine similarity (Singhal, 2001) between ground truth motifs and detected motifs.

The cosine similarity is one for identical and zero for orthogonal patterns.

Not all ground truth motifs extend across all 30 frames, and may have almost vanishing luminosity in the last frames.

Hence, the discovered motifs can be shifted by a few frames and still capture all relevant parts of the motifs.

For this reason we computed the similarity for the motifs with all possible temporal shifts and took the maximum.

More details on the computation of the similarity measure can be found in appendix C.2.We ran both methods on 200 synthetically generated datasets with the parameters shown in table 3 in the appendix.

We here show the results with the correct number of motifs (M = 3) used in both methods.

In appendix E.1 we show that if the number of motifs is overestimated (here M > 3), LeMoNADe still identifies the correct motifs, but they are repeated multiple times in the surplus filters.

Hence this does not reduce the performance of the algorithm.

The temporal extent of the motifs was set to F = 31 to give the algorithms the chance to also capture the longer patterns.

The cosine similarity of the found motifs to the set of ground truth motifs, averaged over all found motifs and all experiments for each of the ten noise levels, is shown in figure 4 .

The results in figure 4 show that LeMoNADe performs as well as SCC in detecting motifs and also shows a similar stability in the presence of noise as SCC.

This is surprising since LeMoNADe does not need the previous extraction of individual cells and hence has to solve a much harder problem than SCC.In order to verify that the results achieved by LeMoNADe and SCC range significantly above chance, we performed a bootstrap (BS) test.

For this, multiple datasets were created with similar spike distributions as before, but with no reoccurring motif-like firing patterns.

We compiled a distribution of similarities between patterns suggested by the proposed method and randomly sampled segments of same length and general statistics from that same BS dataset.

The full BS distributions are shown in appendix C.3.

The 95%-tile of the BS distributions for each noise level are also shown in figure 4.

Figure 5 shows an exemplary result from one of the analysed synthetic datasets with 10% noise and maximum temporal extend of the ground truth motifs of 28 frames.

All three motifs were correctly identified (see figure 5a ) with a small temporal shift.

This shift does not reduce the performance as it is compensated by a corresponding shift in the activations of the motifs (see figure 5b ).

In order to show that the temporal structure of the found motifs matches the ground truth, in figure 5a for motif 1 and 2 we corrected the shift of one and two frames, respectively.

We also show the results after Figure 4: Similarities between found motifs and ground truth for different noise levels.

We show for LeMoNADe (lime green) and SCC (blue) the average similarities between found motifs and ground truth for ten different noise levels ranging from 0% up to 90% spurious spikes.

Error bars indicate the standard deviation.

For each noise level 20 different datasets were analyzed.

For both, LeMoNADe and SCC, the similarities between found and ground truth motifs are significantly above the 95%-tile of the corresponding bootstrap distribution (red) up to a noise level of 70% spurious spikes.

Although LeMoNADe does not need the previous extraction of individual cells, it performs as well as SCC in detecting motifs and also shows a similar stability in the presence of noise.

extracting the individual cells from the motifs and the results from SCC in figure 5c.

One can see that the results are almost identical, again except for small temporal shifts.

We applied the proposed method on two datasets obtained from organotypic hippocampal slice cultures.

The cultures were prepared from 7-9-day-old Wistar rats as described in Kann et al. (2003) and Schneider et al. (2015) .

The fluorescent Ca 2+ sensor, GCaMP6f (Chen et al., 2013) , was delivered to the neurons by an adeno-associated virus (AAV).

Neurons in stratum pyramidale of CA3 were imaged for 6.5 (dataset 1) and 5 minutes (dataset 2) in the presence of the cholinergic receptor agonist carbachol.

For more details on the generation of these datasets see appendix D.1.

motif 0 motif 1 frame 0 motif 2 frame 1 frame 2 frame 3 frame 4 frame 5 frame 6 frame 7 frame 8 frame 9 frame 10 frame 11 frame 12 frame 13 frame 14 frame 15 frame 16 frame 17 frame 18 frame 19 frame 20 frame 0 motif 2 frame 1 frame 2 frame 3 frame 4 frame 5 frame 6 frame 7 frame 8 frame 9 frame 10 frame 11 frame 12 frame 13 frame 14 frame 15 frame 16 frame 17 frame 18 frame 19 frame 20 Figure 6: Result from hippocampal slice culture datasets 1 (top) and 2 (bottom).

The colors in (a) are inverted compared to the standard visualization of calcium imaging data for better visibility.

In (c) activations are thresholded to 70% of the maximum activation for each motif.

In (d) the manually selected frames of motif 0 highlight the temporal structure of the motif.

The proposed method was run on these datasets with the parameter settings shown in table 3 in the appendix E, where we also provide additional comments on the parameter settings.

The analysis of the datasets took less than two hours on a Ti 1080 GPU.

Before running the analysis we computed ∆F/F for the datasets.

We looked for up to three motifs with a maximum extent of F = 21 frames.

The results are shown in figure 6.

For both datasets, one motif in figure 6a consists of multiple cells, shows repeated activation over the recording period (see figure 6b, 6c), and contains temporal structure (see figure 6d) .

The other two "motifs" can easily be identified as artefacts and background fluctuations.

As SCC and many other motif detection methods, LeMoNADe suffers from the fact that such artefacts, especially single events with extremely high neuronal activation, potentially explain a large part of the data and hence can be falsely detected as motifs.

Nevertheless, these events can be easily identified by simply looking at the motif videos or thresholding the activations as done in figure 6c.

Although the found motifs also include neuropil activation, this does not imply this was indeed used by the VAE as a defining feature of the motifs, just that it was also present in the images.

Dendritic/axonal structures are part of the activated neurons and therefore also visible in the motif videos.

If necessary, these structures can be removed by post-processing steps.

As LeMoNADe reduces the problem to the short motif videos instead of the whole calcium imaging video, the neuropil subtraction becomes much more feasible.

We have presented a novel approach for the detection of neuronal assemblies that directly operates on the calcium imaging data, making the cumbersome extraction of individual cells and discrete spike times from the raw data dispensable.

The motifs are extracted as short, repeating image sequences.

This provides them in a very intuitive way and additionally returns information about the spatial distribution of the cells within an assembly.

The proposed method's performance in identifying motifs is equivalent to that of a state-of-the-art method that requires the previous extraction of individual cells.

Moreover, we were able to identify repeating firing patterns in two datasets from hippocampal slice cultures, proving that the method is capable of handling real calcium imaging conditions.

For future work, a post-processing step as used in BID21 or a group sparsity regularization similar to the ones used in BID2 or BID10 could be added to determine a plausible number of motifs automatically.

Moreover, additional latent dimensions could be introduced to capture artefacts and background fluctuations and hence automatically separate them from the actual motifs.

The method is expected to, in principle, also work on other functional imaging modalities.

We will investigate the possibility of detecting motifs using LeMoNADe on recordings from human fMRI or voltage-sensitive dyes in the future.

Variational autoencoder (VAE) are generative latent variable models which were first described in Kingma & Welling (2014) .

The data x = x DISPLAYFORM0 , consisting of N samples of some random variable x, is generated by first drawing a latent variable z (i) from a prior distribution p(z) and then sampling from the conditional distribution p θ * (x | z) with parameters θ * .

The distribution p θ * (x | z) belongs to the parametric family p θ (x | z) with differentiable PDFs w.r.t.

θ and z. Both the true parameters θ * as well as the latent variables z (i) are unknown.

We are interested in an approximate posterior inference of the latent variables z given some data x. The true posterior p θ (z | x), however, is usually intractable.

But it can be approximated by introducing the recognition model (or approximate posterior) q φ (z | x).

We want to learn both the recognition model parameters φ as well as the generative model parameters θ.

The recognition model is usually referred to as the probabilistic encoder and p θ (x | z) is called the probabilistic decoder.

In order to learn the variational parameters φ we want to minimise the KL-divergence between approximate and true posterior KL(q φ (z|x) p θ (z|x)).

Therefore we use the fact that the marginal likelihood p θ (x) can be written as DISPLAYFORM1 As the KL-divergence is non-negative, we can minimize KL q φ (z|x) p θ (z|x) by maximizing the DISPLAYFORM2 In order to optimise the lower bound L(p, q; x) w.r.t.

both the variational parameters φ and the generative parameters θ, we need to compute the gradients DISPLAYFORM3 (10) For the first part of the lower bound the gradient w.r.t.

θ can be easily computed using Monte Carlo sampling DISPLAYFORM4 with z s ∼ q φ (z|x).

The gradient w.r.t.

φ, however, does not take the form of an expectation in z and can therefore not be sampled that easily: DISPLAYFORM5 (12) However, in most cases we can use the reparameterization trick to overcome this problem: the random variablez ∼ q φ (z | x) can be reparameterised using a differentiable transformation h φ (ε, x) of a noise variable ε such thatz = h φ (ε, x) with ε ∼ p(ε) (13) We now can compute the gradient w.r.t.

φ again using Monte Carlo sampling DISPLAYFORM6 with ε s ∼ p(ε).

Hence, the reparameterized lower boundL(p, q; x) ≈ L(p, q; x) can be written as DISPLAYFORM7 with z s = h φ (ε s , x), ε ∼ p(ε).

The first term on the RHS of eq. FORMULA4 is a negative reconstruction error, showing the connection to traditional autoencoders, while the KL-divergence acts as a regularizer on the approximate posterior q φ (z | x).

The encoder network starts with a few convolutional layers with small 2D filters operating on each frame of the video separately, inspired by the architecture used in BID1 to extract cells from calcium imaging data.

The details of this network are shown in table 1.

Afterwards the feature maps of the whole video are passed through a final convolutional layer with 3D filters.

These filters have size of the feature maps gained from the single images times a temporal component of length F , which is the expected maximum temporal extent of a motif.

We use 2 · M filters and apply padding in the temporal domain to avoid edge effects.

By this also motifs that are cut off at the beginning or the end of the sequence can be captured properly.

The output of the encoder are 2 · M feature maps of size (T + F − 1) × 1 × 1.

Instead of reparameterizing the Bernoulli distributions, we will reparameterize their BinConcrete relaxations.

The BinConcrete relaxation of a Bernoulli distribution with parameter α takes as input parameterα = α/(1 − α).

BID11 showed that instead of using the normalized probabilities α, we can also perform the reparametrization with unnormalized parameters α 1 and α 2 , where α 1 is the probability to sample a one and α 2 is the probability to sample a zero andα = α 1 /α 2 .The first M feature maps, which were outputted by the encoder, are assigned to contain the unnormalised probabilities α for all m = 1, . . .

, M and t = 1, . . .

, T + F − 1.

The multiplication by α 1 m,t in eq. FORMULA7 is not part of the original reparametrization trick BID11 Jang et al., 2017 ).

But we found that the results of the algorithm improved dramatically as we scaled the activations with the α 1 -values that were originally predicted from the encoder network.

The input to the decoder are now the activations z. The decoder consists of a single deconvolution layer with M filters of the original frame size times the expected motif length F .

These deconvolution filters contain the motifs we are looking for.

The details of the used networks as well as the sizes of the inputs and outputs of the different steps are shown in table 1.

Algorithm 1 summarizes the reparametrization and updates.

We created 200 artificial sequences of length 60 s with a frame rate of 30 fps and 128 × 128 pixel per image.

The number of cells was varied and they were located randomly in the image plane with an overlap of up to 30 %.

The cell shapes were selected randomly from 36 shapes extracted from Step x sub ← decode via f θ (z) // Update Parameters Compute gradients of loss φ, θ ← update via ∇ φ,θ (x sub , x sub ,α, λ 1 ,ã, λ 2 , β KL ) (see eq. FORMULA7 in the main paper) until until convergence of θ, φ; real data.

The transients were modelled as two-sided exponential decay with scales of 50 ms and 400 ms, respectively.

In contrast to Diego & Hamprecht (2013), we included neuronal assemblies with temporal firing structure.

That means cells within an assembly can perform multiple spikes in a randomly chosen but fixed motif of temporal length up to 30 frames.

We used 3 different assemblies in each sequence.

The assembly activity itself was modelled as a Poisson process (Lopes-dos Santos et al., 2013) with a mean of 0.15 spikes/second and a refractory period of at least the length of the motif itself.

By construction the cell locations as well as the firing motifs are known for these datasets.

In order to simulate the conditions in real calcium imaging videos as good as possible, we added Gaussian background noise with a relative amplitude (max intensity − mean intensity)/σ noise between 10 and 20.

Additionally, spurious spikes not belonging to any motif were added.

The amount of spurious spikes was varied from 0% up to 90% of all spikes in the dataset.

For each of the 10 noise levels 20 datasets were generated.

DISPLAYFORM0

The performance of the algorithms is measured by computing the cosine similarity (Singhal, 2001) between ground truth motifs and found motifs.

The found motifs are in an arbitrary order, not necessarily corresponding to the order of the ground truth motifs.

Additionally, the found motifs can be shifted in time compared to the ground truth.

To account for this fact, we compute the similarity between the found motifs and each of the ground truth motifs with all possible temporal shifts and take the maximum.

Hence, the similarity between the m-th found motif and the set of ground truth motifs G is defined by DISPLAYFORM0 where M m is the m-th found motif, ·, · is the dot product and vec(·) vectorizes the motifs with dimensions F × N into a vector of length F · N , where N is the number of cells.

The shift operator s→ (·) moves a motif s frames forward in time while keeping the same size and filling missing values appropriately with zeros (Smaragdis, 2004) .The cosine similarity of the found motifs to the set of ground truth motifs was averaged over all found motifs and all experiments for each noise level.

The average similarities achieved with LeMoNADe and SCC as well as the 5% significance threshold of the BS distribution for each noise level can be found in table 2.

Statistical methods for testing for cell assemblies (or spatio-temporal patterns more generally) have been advanced tremendously in recent years, addressing many of the issues that have plagued older approaches (Grün, 2009; Staude et al., 2010a; BID0 BID26 .

Simple shuffle bootstraps are not necessarily the best methods if they destroy too much of the auto-correlative structure, and they can severely underestimate the distributional tails (Davison et al., 1997) .

Therefore we use sophisticated parametric, model-based bootstraps which retain the full statistical structure of the original data, except for the crucial feature of repeating motifs.

In order to provide a 'null hypothesis (H0)' reference for the motif similarities returned by LeMoNADe (or other methods), we used the following bootstrap (BS) based test procedure: We generated 20 datasets analogue to those described in section C.1, i.e. with same spiking statistics and temporal Figure 7: Top: Bootstrap distribution for similarity between random patterns.

Shown is a sample from the BS distribution (blue) and the 95% significance threshold (red).

Bottom: Distribution for similarity between patterns found on data which contained repeating motifs.

Shown are the similarities between motifs found with LeMoNADe (lime green) and the ground truth motifs for the synthetic datasets discussed in the paper, which contained repeating motifs.

The 95% significance threshold of the corresponding BS distribution is indicated as vertical red line.

convolution with calcium transients, but without repeating motifs.

These motif-less H0 datasets were then processed by LeMoNADe in the very same way as the motif-containing datasets, i.e. with the parameter settings as shown in table 3.

From each of these BS datasets 150 random samples of the same temporal length as that of the 'detected' motifs were drawn.

For each BS dataset, the similarities between each of the found motifs and all of the 150 random samples were computed as described in section C.2.

As datasets with higher noise levels have different spiking statistics, we repeated this procedure for each of the ten noise levels.

Figure 7 shows the BS distributions (top).

We also show the distribution of similarities between motifs found with LeMoNADe on the datasets which contained motifs (bottom).

The 95%-tile (corresponding to a 5% alpha level) of the BS distribution is displayed as vertical red line.

Up to a noise level of 70% the average of the similarities found on the datasets that contained motifs is much higher than the 95%-tile of the BS distribution.

Organotypic hippocampal slice cultures were prepared from 7-9-day-old Wistar rats (Charles River Laboratories, Sulzfeld, Germany) as described by Kann et al. (2003) and Schneider et al. (2015) .

Animals were taken care of and handled in accordance with the European directive 2010/63/EU and with consent of the animal welfare officers at Heidelberg University (license, T96/15).Slices were infected with adeno-associated virus (AAV) obtained from Penn Vector Core (PA, USA) encoding GCaMP6f under the control of the CamKII promoter AAV5.CamKII.GCaMPf.

WPRE.SV40, Lot # V5392MI-S).

AAV transduction was achieved, under sterile conditions, by applying 0.5µl of the viral particles solution (qTiter: 1.55e13 GC/ml) on top of the slices.

Slices were maintained on Biopore membranes (Millicell standing inserts; Merck Millipore, Schwalbach, Germany) between culture medium.

The medium consisted of 50% minimal essential medium, 25% Hank's balanced salt solution (Sigma-Aldrich, Taufkirchen, Germany), 25% horse serum (Life Technologies, Darmstadt, Germany), and 2mM L-glutamine (Life Technologie) at pH 7.3, stored in an incubator (Heracell; Thermoscientific, Dreieich, Germany) with humidified normal atmosphere (5% CO2, 36.5• C).

The culture medium (1 ml) was replaced three times per week.

Artificial cerebrospinal fluid used for imaging was composed of 129 mM NaCl, 3 mM KCl, 1.25 mM NaH2PO4, 1.8 mM MgSO4, 1.6 mM CaCl2, 21 mM NaHCO3, and10 mM glucose (Sigma-Aldrich, Taufkirchen, Germany).

The pH of the recording solution was 7.3 when it was saturated with the gas mixture (95% O2, 5% CO2).

Recording temperature was 32 ± 1 • C. Constant bath wash of 20µM (dataset 1) and 10µM (dataset 2) carbachol (Sigma-Aldrich) was performed to enhance neuronal activity and increase firing probability during imaging BID15 (b) Difference between motif 0 found on real dataset 2 and the constructed synchronous firing pattern.

Figure 8: Color-coded difference between discovered motifs and intensity modulated synchronous firing.

Red color indicates negative differences, blue positive differences and white zero difference.

The fact that for both datasets in motif 0 some cells are displayed in red over multiple frames shows that these motifs contain temporal structure beyond mere spiking synchrony.

Imaging of CA3 region of the hippocampus was performed on day 29 with 20x magnification (dataset 1) and on day 30 with 10x magnification (dataset 2) in vitro (23 days post viral infection) from slices maintained in submerged chamber of Olympus BX51WI microscope.

GCaMP6f was excited at 485 ± 10nm.

Fluorescence images (emission at 521 ± 10nm) were recorded at 6.4Hz (dataset 1) and 4Hz (dataset 2) using a CCD camera (ORCA-ER; Hamamatsu Photonics, Hamamatsu City, Japan).

Before running the analysis we computed ∆F/F for the datasets.

In order to perform the computations more efficiently, we cropped the outer parts of the images containing no interesting neuronal activity and downsampled dataset 2 by a factor of 0.4.

In order to show that the motifs 0 found in the two real datasets contain temporal structure, we compare them to what the synchronous activity of the participating cells with modulated amplitude would look like.

The synchronous firing pattern was constructed as follows: First, for the motif M m with m = 1, . . .

, M the maximum projection P m at each pixel p = 1, . . .

, P · P over time was computed by DISPLAYFORM0 and normalizedP DISPLAYFORM1 Finally, the synchronous firing pattern S m for motif m is gained by multiplying this normalized maximum projection at each time frame f with the maximum intensity of motif m at that frame: DISPLAYFORM2 Figures 8 shows the difference between the found motif and the constructed synchronous firing patterns for the motifs found on the two real datasets.

In order to show that LeMoNADe performs similar to SCC not only on synthetically generated data but also on real data, we ran both methods on real dataset 2.

A well trained neuroscientist manually extracted the individual cells and calcium traces from the original calcium imaging video.

FIG8 shows the result obtained with SCC on these traces.

In the same manner calcium traces were extraced from the motif found with LeMoNADe (see FIG8 ).

Both results in figure 9 are highly similar.

LeMoNADe is not more difficult to apply than other motif detection methods for neuronal spike data.

In our experiments, for most of the parameters the default settings worked well on different datasets and only three parameters need to be adjusted: the maximum number of motifs M , the maximum 10 1 motif length F , and one of the sparsity parameters (e.g.ã or β KL ).

For SCC the user also has to specify three similar parameters.

In addition, SCC requires the previous extraction of a spike matrix which implies many additional parameters.

TAB5 shows the parameter settings used for the experiments shown in the paper.

In order to show the effects of over-and underestimating the number of motifs, we first use our synthetic data with existing ground truth and 3 true motifs and run LeMoNADe with underestimated (M = 1), correct (M = 3) and overestimated (M = 5) number of expected motifs.

FIG4 shows the complete ground truth (figure 10a) and found motifs for the exemplary synthetic dataset discussed in the paper.

Besides the results for M = 3 (figure 10c) we also show the found motifs for M = 1 (figure 10b) and M = 5 ( FIG4 .

If the number of motifs is underestimated (M = 1) only one of the true motifs is captured.

When the number of motifs is overestimated (M = 5) the correct motifs are identified and the surplus filters are filled with (shifted) copies of the true motifs and background noise.

We also investigated the influence of different numbers of motifs on the results on real datasets.

FIG4 shows the found motifs on dataset 1 for the different numbers of motifs M = 1, 2, 3, 5.

When the number is limited (as for M = 1), the model is expected to learn those motifs first which best explain the data.

The motif shown in figure 11a also appears if M is increased.

This shows that this motif is highly present in the data.

However, as long as only one filter is available the motif also contains a lot of background noise.

The second filter in FIG4 contains a high luminosity artefact of the data.

With its high luminosity and large spacial extent, it explains a lot of the dataset.

However, it can easily be identified as no neuronal assembly.

If the number of motifs is further increased to M = 3 (see FIG4 ), more background noise is captured in the additional filter and the motif becomes cleaner.

When the number of motifs is further increased to M = 5, no new motifs appear and the surplus two filters seem to be filled up with parts of the structures which were already present in 11c.

Hence, when the correct number of motifs is unknown (as expected for real datasets) we recommend to slightly overestimate the expected number of motifs.

The result will capture the true motifs plus some copies of them.

In future work, a post-processing step as in BID21 or a group sparsity regularization as in BID2 and BID10 could be introduced to eliminate these additional copies automatically.

Background noise could be easily identified as no motif by either looking at the motif videos or thresholding the found activations.

In future extends of the model we will study the effect of additional latent dimensions for background noise to automatically separate it from actual motifs.

If the maximum motif length F is underestimated the found motifs are expected to just contain the part of the motif that reduces the reconstruction error most.

Hence in most cases the most interesting parts of the motifs will be captured but details at either end of the motifs could be lost.

If the motif length is overestimated, the motifs can be captured completely but might be shifted in time.

This shift, however, will be compensated by the motif activations and hence has no negative effect on the results.

In our experiments we achieved good results with a generously chosen motif length.

For this reason we recommend to overestimate the motif length.

FIG4 shows the found motifs on real dataset 1 with M = 3 and for the different motif lengths F = 21 and F = 31.

The results are highly similar.

In both cases, the interesting pattern (motif 0 in FIG4 and motif 1 in figure 12b, respectively) is captured.

The parameterã influences the sparsity of the found activations.

Smaller values ofã will penalize activations harder and hence often result in cleaner and more meaningful motifs.

However, ifã is too small it will suppress the activations completely.

For this reason we recommend to perform for each new dataset experiments with different values ofã.

Changing the value of β KL is another option to regulate the sparsity of the activations.

However, in our experiments we found that the default value of β KL = 0.1 worked well for many different datasets and varyingã was effective enough.

For the temperature parameters the default values λ 1 = 0.6 and λ 2 = 0.5 worked well in most cases and changing them is usually not necessary.

In order to show the reaction of the method to the choice ofã and β KL we performed multiple experiments on the real dataset 2 with different parameter settings.

We fixed all parameters as shown in table 3 except forã (figures 13 and 14) and β KL (figures 15 and 16).Whenã is varied within one order of magnitude (see FIG4 ) the motifs look quite similar -except for temporal shifts of the motifs and shuffling of the order of the motifs.

For smaller values ofã surplus filters are filled with background noise (see figures 13a to 13d), whereas for a bit larger values ofã the surplus filters are filled with copies of (parts of) the motif (see figures 13e to 13g).

Note that the motif which was also highlighted in the paper (figure 6d) appears in all results from figure 13b to 13g at least once.

Only ifã is changed by more than one order of magnitude the results become significantly different and the motif is no longer detected (see FIG4 .

This indicates that it is sufficient to vary only the order of magnitude ofã in order to find a regime where motifs appear in the results and fine tuningã is not necessary.

This strategy is also the recommended strategy to find an appropriate sparsity parameter in SCC.A similar behavior can be observed when β KL is varied (see FIG4 for changes within an order of magnitude and FIG4 for larger changes).

One can see similar effects as for the variation ofã, but in the opposite direction: for smaller β KL surplus filters are rather filled with copies of the motif whereas for larger values of β KL the surplus filters are filled with background noise.

This shows that it is usually sufficient to only tune one of the two -eitherã or β KL -in order to achieve good results.motif 0 motif 1 frame 0 motif 2 frame 1 frame 2 frame 3 frame 4 frame 5 frame 6 frame 7 frame 8 frame 9 frame 10 frame 11 frame 12 frame 13 frame 14 frame 15 frame 16 frame 17 frame 18 frame 19 frame 20 frame 21 frame 22 frame 23 frame 24 frame 25 frame 26 frame 27(a) Ground truth motifs frame 0 motif 0 frame 1 frame 2 frame 3 frame 4 frame 5 frame 6 frame 7 frame 8 frame 9 frame 10 frame 11 frame 12 frame 13 frame 14 frame 15 frame 16 frame 17 frame 18 frame 19 frame 20 frame 21 frame 22 frame 23 frame 24 frame 25 frame 26 frame 27 frame 28 frame 29 frame 30 (b) Found motifs for M = 1 motif 0 motif 1 frame 0 motif 2 frame 1 frame 2 frame 3 frame 4 frame 5 frame 6 frame 7 frame 8 frame 9 frame 10 frame 11 frame 12 frame 13 frame 14 frame 15 frame 16 frame 17 frame 18 frame 19 frame 20 frame 21 frame 22 frame 23 frame 24 frame 25 frame 26 frame 27 frame 28 frame 29 frame 30 (c) Found motifs for M = 3 motif 0 motif 1 motif 2 motif 3 frame 0 motif 4 frame 1 frame 2 frame 3 frame 4 frame 5 frame 6 frame 7 frame 8 frame 9 frame 10 frame 11 frame 12 frame 13 frame 14 frame 15 frame 16 frame 17 frame 18 frame 19 frame 20 frame 21 frame 22 frame 23 frame 24 frame 25 frame 26 frame 27 frame 28 frame 29 frame 30 frame 0 motif 0 frame 1 frame 2 frame 3 frame 4 frame 5 frame 6 frame 7 frame 8 frame 9 frame 10 frame 11 frame 12 frame 13 frame 14 frame 15 frame 16 frame 17 frame 18 frame 19 frame 20 (a) Found motif for M = 1 motif 0 frame 0 motif 1 frame 1 frame 2 frame 3 frame 4 frame 5 frame 6 frame 7 frame 8 frame 9 frame 10 frame 11 frame 12 frame 13 frame 14 frame 15 frame 16 frame 17 frame 18 frame 19 frame 20 (b) Found motifs for M = 2 motif 0 motif 1 frame 0 motif 2 frame 1 frame 2 frame 3 frame 4 frame 5 frame 6 frame 7 frame 8 frame 9 frame 10 frame 11 frame 12 frame 13 frame 14 frame 15 frame 16 frame 17 frame 18 frame 19 frame 20 (c) Found motifs for M = 3 motif 0 motif 1 motif 2 motif 3 frame 0 motif 4 frame 1 frame 2 frame 3 frame 4 frame 5 frame 6 frame 7 frame 8 frame 9 frame 10 frame 11 frame 12 frame 13 frame 14 frame 15 frame 16 frame 17 frame 18 frame 19 frame 20 motif 0 motif 1 frame 0 motif 2 frame 1 frame 2 frame 3 frame 4 frame 5 frame 6 frame 7 frame 8 frame 9 frame 10 frame 11 frame 12 frame 13 frame 14 frame 15 frame 16 frame 17 frame 18 frame 19 frame 20(a) Found motifs for F = 21 motif 0 motif 1 frame 0 motif 2 frame 1 frame 2 frame 3 frame 4 frame 5 frame 6 frame 7 frame 8 frame 9 frame 10 frame 11 frame 12 frame 13 frame 14 frame 15 frame 16 frame 17 frame 18 frame 19 frame 20 frame 21 frame 22 frame 23 frame 24 frame 25 frame 26 frame 27 frame 28 frame 29 frame 30 (b) Found motifs for F = 31 analyzed data and reconstructed videos at https://drive.google.com/drive/folders/ 19F76JLn490RzZ4d7GxbWZoq6RdF2nt3w?usp=sharing.

The reconstructed videos are gained by convolving the found motifs with the corresponding found activations.

The videos are provided either in TIFF or MP4 format.

TAB6 shows the names of the files together with short descriptions what each video shows.

The videos corresponding to the synthetic dataset were generated with a frame rate of 30 fps and those corresponding to the real dataset with 10 fps. ) and reconstructions from the found motifs; and RGB video showing a superposition of RGB values of the reconstructed videos from the three motifs found on the dataset.

Additionally to the synthetic data example discussed in the paper (with 10% noise spikes), we also provide videos from a synthetic dataset with 50% spurious spikes.

@highlight

We present LeMoNADe, an end-to-end learned motif detection method directly operating on calcium imaging videos.

@highlight

This paper proposes a VAE-style model for identifying motifs from calcium imaging videos, relying on Bernouli variables and requires Gumbel-softmax trick for inference.