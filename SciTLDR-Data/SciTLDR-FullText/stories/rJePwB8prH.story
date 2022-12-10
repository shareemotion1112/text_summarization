Disentangled encoding is an important step towards a better representation learning.

However, despite the numerous efforts, there still is no clear winner that captures the independent features of the data in an unsupervised fashion.

In this work we empirically evaluate the performance of six unsupervised disentanglement approaches on the mpi3d toy dataset curated and released for the NeurIPS 2019 Disentanglement Challenge.

The methods investigated in this work are Beta-VAE, Factor-VAE, DIP-I-VAE, DIP-II-VAE, Info-VAE, and Beta-TCVAE.

The capacities of all models were progressively increased throughout the training and the hyper-parameters were kept intact across experiments.

The methods were evaluated based on five disentanglement metrics, namely, DCI, Factor-VAE, IRS, MIG, and SAP-Score.

Within the limitations of this study, the Beta-TCVAE approach was found to outperform its alternatives with respect to the normalized sum of metrics.

However, a qualitative study of the encoded latents reveal that there is not a consistent correlation between the reported metrics and the disentanglement potential of the model.

Unsupervised disentanglement is an open problem in the realm of representation learning, incentivized around interpretability BID8 BID1 .

A disentangled representation is a powerful tool in transfer learning, few shot learning, reinforcement learning, and semi-supervised learning of downstream tasks (Goo, 2018; BID9 BID1 .Here, we investigate the performance of some of the promising disentanglement methods from the family of variational autoencoders (VAE).

The methods are evaluated based on five relatively established disentanglement metrics on the simplistic rendered images of the mpi3d toy dataset curated and released for the NeurIPS 2019 Disentanglement Challenge.

To mitigate the sensitivity of the models to the initial state, as suggested by the findings of Goo (2018), an autoencoder model was pre-trained with the conventional VAE objective BID6 on the mpi3d toy dataset.

This approach guaranteed that models did not collapse into a local minima with little to no reconstruction.

It also facilitated the training process given the constraints on the length of training by the challenge.

In this preliminary study, we implemented the variational objective functions proposed by the following methods: β-VAE BID4 ), β-TCVAE (Chen et al., 2018 , Factor-VAE BID5 , Info-VAE BID11 , DIP-I-VAE, and DIP-II-VAE BID7 .In β-TCVAE, the mutual information between the data variables and latent variables are maximized, while the mutual information between the latent variables are minimized.

Defining x n as the nth sample of the dataset, the evidence lower bound (ELBO) of this objective can be simplified as follows 1 DISPLAYFORM0 where z j denotes the jth dimension of the latents.

In the above equation, the first term is the reconstruction loss.

The second term is the distance between the assumed prior distribution of the latent space and the empirical posterior latent distribution.

The last term is an indication of the total correlation (TC) between the latent variables which is a generalization of the mutual information for more than two variables BID10 .

A total capacity constraint which limits the KL divergence between the posterior latent distribution and the factorized prior can encourage the latent representation to be more factorised.

However, this will act as an information bottleneck for the reconstruction task and results in a blurry reconstruction.

Thus, progressively increasing the information capacity of VAE during training can help facilitate the robust learning of the factorized latents BID2 .

This is achieved by introducing the capacity term C and defining the distance between distributions as the absolute deviation from C: DISPLAYFORM0 Gradually increasing C has an annealing effect on the constraint and increases the reconstruction capacity of the model.

For each learning algorithm, the hyper-parameter sub-spaces were independently searched.

However, in order for the results reported here to be comparable, the hyper-parameters were kept intact in between the following experiments.

The input images were 64 × 64 pixels and the latent space was of size 20.

The model capacity parameter, C, was initiated at zero and gradually increased up to 25 over 2000 iterations.

Learning rate was initiated at 0.001 and was reduced by a factor of 0.95 when the loss function (Equation (1)) did not decrease after two consecutive epochs, down to a minimum of 0.0001 .

Batch size was set to 64.

Optimization was carried out using the Adam optimizer with the default parameters β1 = 0.9 and β2 = 0.999.

The network architectures and other hyper-parameters are detailed in Appendix A.The trained models were evaluated based on five evaluation metrics, namely, DCI, FactorVAE metric, IRS, MIG, and SAP-Score.

Results of these evaluations are presented in TAB0 .

The non-ignored latent variables of each method are traversed and the results are visualized in Appendix B. Moreover, the evaluation logs during model training are visualized in Appendix C.All the models and experiments were implemented using the PyTorch deep learning library and packaged under the Disentanglement-PyTorch repository https://github.com/ amir-abdi/disentanglement-pytorch 2 .

In this work we compared the degree of disentanglement in latent encodings of six variational learning algorithms, namely, β-VAE, Factor-VAE, DIP-I-VAE, DIP-II-VAE, Info-VAE, and β-TCVAE.

The empirical results TAB0 point to β-TCVAE being marginally the superior option and, consequently, chosen as the best performing approach.

However, a qualitative study of the traversed latent spaces (Appendix B) reveals that none of the models encoded a true disentangled representation.

Lastly, although the DIP-VAE-II model is under performing according to the quantitative results, it has the least number of ignored latent variables with a promising latent traversal compared to other higher performing methods (Appendix B).

As a result of these inconsistencies, we find the five metrics utilized in this study inadequate for the purpose of disentanglement evaluation.

Among the limitations of this study is the insufficient search of the hyper-parameters space for all the six learning algorithms.

Moreover, the NeurIPS 2019 Disentanglement Challenge imposed an 8-hour limit on the training time of the models which we found to be insufficient.

This, while the maximum number of iterations was set to 200k in our experiments, this value was limited to 100k in the submissions made to the challenge portal.2.

The repository will be publicly released upon the completion of the competition.

The encoder neural network in all experiments consisted of 5 convolutional layers with strides of 2, kernel sizes of 3 × 3, and number of kernels gradually increasing from 32 to 256.

The encoder ended with a dense linear layer which estimated the posterior latent distribution as a parametric Gaussian.

The decoder network consisted of one convolutional followed with 6 deconvolutional (transposed convolutional) layers, with kernel sizes of 4, strides of 2, and the number of kernels gradually decreasing from 256 down to the number of channels of the image space.

ReLU activations were used throughout the architecture, except for the last layers of the encoder and decoder networks.

@highlight

Inadequacy of Disentanglement Metrics