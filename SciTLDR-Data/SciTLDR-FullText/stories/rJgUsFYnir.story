Unsupervised learning of disentangled representations is an open problem in machine learning.

The Disentanglement-PyTorch library is developed to facilitate research, implementation, and testing of new variational algorithms.

In this modular library, neural architectures, dimensionality of the latent space, and the training algorithms are fully decoupled, allowing for independent and consistent experiments across variational methods.

The library handles the training scheduling, logging, and visualizations of reconstructions and latent space traversals.

It also evaluates the encodings based on various disentanglement metrics.

The library, so far, includes implementations of the following unsupervised algorithms VAE, Beta-VAE, Factor-VAE, DIP-I-VAE, DIP-II-VAE, Info-VAE, and Beta-TCVAE, as well as conditional approaches such as CVAE and IFCVAE.

The library is compatible with the Disentanglement Challenge of NeurIPS 2019, hosted on AICrowd and was used to compete in the first and second stages of the challenge, where it was ranked among the best few participants.

There are two overlapping avenues in representation learning.

One focuses on learning task-specific transformations often optimized towards specific domains and applications.

The other approach learns the intrinsic factors of variation, in a disentangled and taskinvariant fashion.

The unsupervised disentanglement of latent factors, where changes in a single factor of variation shifts the latent encoding in a single direction, is an open problem of representation learning (Bengio et al., 2013; Lake et al., 2016) .

Disentangled representations are valuable in few-shot learning, reinforcement learning, transfer learning, as well as semisupervised learning (Bengio et al., 2013; Peters et al., 2017; .

In this work, we developed a library based on the functionalities of the PyTorch framework, which facilitates research, implementation, and testing of new variational algorithms focusing on representation learning and disentanglement.

The library branches from the Disentanglement Challenge of NeurIPS 2019, hosted on AICrowd (aicrowd.com), and was used to compete in the first and second stages of the challenge where it was highly ranked.

The Disentanglement-PyTorch library is released under the GNU General Public License at https://github.com/amir-abdi/disentanglement-pytorch.

Unsupervised Objectives Currently, the library includes implementations of the following unsupervised variational algorithms: VAE (Kingma and Welling, 2014), ??-VAE (Fertig et al., 2018) , ??-TCVAE (Chen et al., 2018), Factor-VAE (Kim and Mnih, 2018) , Info-VAE (Zhao et al., 2017) , DIP-I-VAE, and DIP-II-VAE (Kumar et al., 2018) .

Algorithms are implemented as plug-ins to the variational Bayesian formulation, and are specified by the loss terms flag.

As a result, if the loss terms of two learning algorithms (e.g., A and B) were found to be compatible, they can both be included in the objective function with the flag set as [--loss terms A B].

This enables researchers to mix and match loss terms which optimize towards correlated goals.

The library supports conditional approaches such as CVAE (Sohn et al., 2015) , where extra known attributes (i.e, labels) are included in the encoding and decoding processes.

It also supports IFCVAE, inspired by the IFcVAE-GAN (Creswell et al., 2017) , which enforces certain latent factors to encode known attributes using a set of positive (auxiliary) and negative (adversarial) discriminators in a supervised fashion.

Thanks to the modular implementation of the library, any of the above-mentioned unsupervised loss terms can be used with conditional and information factoriation approaches to encourage disentanglement across attribute-invariant latents.

Neural architectures and the dimensionality of the data and the latent spaces are configurable and decoupled from the training algorithm.

Consequently, new architectures for the encoder and decoder networks, such as the auto-regressive models, and support for other data domains, can be independently investigated.

We rely on Google's implementation of the disentanglement metrics to evaluate the quality of the learned representations .

Thanks to the disentanglement-lib 1 library, the following metrics are currently supported: BetaVAE (Higgins et al., 2017) , FactorVAE (Kim and Mnih, 2018) , Mutual Information Gap (MIG) (Chen et al., 2018) , Interventional Robustness Score (IRS) (Suter et al., 2019) , Disentanglement Completeness and Informativeness (DCP) (Eastwood and Williams, 2018) , and Separated Attribute Predictability (SAP) (Kumar et al., 2018) .

Controlled Capacity Increase It is shown that gradually relaxing the information bottleneck during training improves the disentanglement without penalizing the reconstruction accuracy.

Following the formulation of Burgess et al. (2018) , the capacity, defined as the distance between the prior and the latent posterior distributions and denoted with the variable C, is gradually increased during training.

To avoid convergence points with high reconstruction loss, training can be started with more emphasis on the reconstruction and gradually relaxing for the disentanglement term to become more relative.

Dynamic Learning Rate Scheduling All forms of learning rate schedulers are supported.

Researchers are encouraged to leverage the dynamic LR scheduling to gradually decrease the rate when the average objective function over the epoch stops its decremental trend.

Logging and Visualization The library leverages the Weights & Biases (W&B) 2 tool to record and visualize the training process and experiments' results.

Besides the scalar values, we visualize the attribute and condition traversals, latent factor traversals, and input reconstructions, both as static images (logged via W&B) as well as animated GIFs.

The ??-TCVAE algorithm achieved the best disentanglement results on the mpi3d real dataset in the second stage of the disentanglement challenge.

Given the limited 8-hour training time for the challenge, the model was pre-trained on the mpi3d toy dataset (Gondal et al., 2019) .

The model was trained with the Adam optimizer for 90k iterations on batches of size 64.

The ?? value of the ??-TCVAE objective function was set to 2.

The learning rate was initialized at 0.001 and reduced on the plateau of the objective function with a factor of 0.95.

The capacity parameter, C, was gradually increased from 0 to 25.

The dimensionality of the z space was generously set to 20.

The encoder consisted of 5 convolutional layers with strides of 2, kernel sizes of 3 ?? 3, and number of kernels gradually increasing from 32 to 256.

The encoder ended with a dense linear layer which estimated the posterior latent distribution as a parametric Gaussian.

The decoder network consisted of one convolutional followed with 6 deconvolutional (transposed convolutional) layers, with kernel sizes of 4, strides of 2, and the number of kernels gradually decreasing from 256 down to the number of channels of the image space.

ReLU activations were used except for the last layers of the encoder and decoder networks.

Model's performance on the unseen objects of mpi3d realistic and mpi3d real datasets are presented in Table 1 .

The configurations of the two experiments are the same and the model consistently performed better on the mpi3d real dataset.

This was unexpected as the model was initialized on the mpi3d toy dataset.

Disentanglement performance, on the available samples of the mpi3d realistic dataset, is visualized in Appendix A. Figure 1 : Latent factor traversal of the trained ??-TCVAE model on a random sample of the mpi3d realistic dataset.

As demonstrated, the disentanglement is not complete and some features are encoded in the same latent factor.

A latent space of size 20 was used, however, changes in the other 13 latent factors had no effect on the reconstruction; thus, these feature-invariant factors were not included for brevity.

@highlight

Disentanglement-PyTorch is a library for variational representation learning