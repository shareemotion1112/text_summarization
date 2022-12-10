The challenge of learning disentangled representation has recently attracted much attention and boils down to a competition.

Various methods based on variational auto-encoder have been proposed to solve this problem, by enforcing the independence between the representation and modifying the regularization term in the variational lower bound.

However recent work by Locatello et al. (2018) has demonstrated that the proposed methods are heavily influenced by randomness and the choice of the hyper-parameter.

This work is built upon the same framework in Stage 1 (Li et al., 2019), but with different settings; to make it self-contained, we provide this manuscript, which is unavoidably very similar to the report for Stage 1.

In detail, in this work, instead of designing a new regularization term, we adopt the FactorVAE but improve the reconstruction performance and increase the capacity of network and the training step.

The strategy turns out to be very effective in achieving disentanglement.

The great success of unsupervised learning heavily depends on the representation of the feature in the real-world.

It is widely believed that the real-world data is generated by a few explanatory factors which are distributed, invariant, and disentangled (Bengio et al., 2013) .

The challenge of learning disentangled representation boils down into a competition 1 to build the best disentangled model.

The key idea in disentangled representation is that the perfect representation should be a one-to-one mapping to the ground truth disentangled factor.

Thus, if one factor changed and other factors fixed, then the representation of the fixed factor should be fixed accordingly, while others' representation changed.

As a result, it is essential to find representations that (i) are independent of each other, and (ii) align to the ground truth factor.

Recent line of works in disentanglement representation learning are commonly focused on enforcing the independence of the representation by modifying the regulation term in the (Kumar et al., 2018) and FactorVAE (Kim and Mnih, 2018) .

See Appendix A for more details of these model.

To evaluate the performance of disentanglement, several metrics have been proposed, including the FactorVAE metric (Kim and Mnih, 2018) , Mutual Information Gap (MIG) (Chen et al., 2018) , DCI metric (Eastwood and Williams, 2018) , IRS metric (Suter et al., 2019) , and SAP score (Kumar et al., 2018) .

However, one of our findings is that these methods are heavily influenced by randomness and the choice of the hyper-parameter.

This phenomenon was also discovered by Locatello et al. (2018) .

Therefore, rather than designing a new regularization term, we simply use FactorVAE but at the same time improve the reconstruction performance.

We believe that, the better the reconstruction, the better the alignment of the ground-truth factors.

Therefore, the more capacity of the encoder and decoder network, the better the result would be.

Furthermore, after increasing the capacity, we also try to increase the training step which also shows a significant improvement of evaluation metrics.

The final architecture of FactorVAE is given in Figure 1 .

Note that, this work is built upon the same framework in stage 1 (Li et al., 2019) , but with different settings; to make it self-contained, we provide this manuscript, which is unavoidably very similar to the report for Stage 1.

Overall, our contribution can be summarized as follow: (1) we found that the performance of the reconstruction is also essential for learning disentangled representation, and (2) we achieve state-of-the-art performance in the competition.

In this section, we explore the effectiveness of different disentanglement learning models and the performance of the reconstruction for disentangle learning.

We first employ different kinds of variational autoencoder including BottleneckVAE, AnneledVAE, DIPVAE, BetaTCVAE, and BetaVAE with 30000 training step.

Second, we want to know whether the capacity plays an important role in disentanglement.

The hypothesis is that the larger the capacity, the better reconstruction can be obtained, which further reinforces the disentanglement.

In detail, we control the number of latent variables.

In this section, we present our experiment result in stage 1 and stage 2 of the competition.

We first present the performance of different kinds of VAEs in stage 1, which is given in Table 1 .

It shows that FactorVAE achieves the best result when the training step is 30000.

In the following experiment, we choose FactorVAE as the base model.

Then, as shown in Table 2 , we increase the step size and we find that the best result was achieved at 1000k training steps.

The experiment in this part may not be sufficient, but it still suggests the fact that the larger the capacity is, the better the disentanglement performance.

Since we increase the capacity of the model, it is reasonable to also increase the training steps at the same time.

Furthermore, as shown in Table 3 , using sufficient large training step (≥ 800k), we investigate the effectiveness of the number of latent variables.

This experiment is performed in stage 2 and suggests that the FactorVAE and the DCI metric are positive as the latent variables increase, while the other metrics decrease.

The best result in the ranking is marked as bold, which suggests that we should choose an appropriate number of latent variables.

In this work, we conducted an empirical study on disentangled learning.

We first conduct several experiments with different disentangle learning methods and select the FactorVAE as the base model; and second we improve the performance of the reconstruction, by increasing the capacity of the model and the training step.

Finally, our results appear to be competitive.

(VAE) (Kingma and Welling, 2013 ), a generative model that maximize the following evidence lower bound to approximate the intractable distribution p θ (x|z) using q φ (z|x),

where q φ (z|x) denote Encoder with parameter φ and p θ (x|z) denote Decoder with parameter θ.

As shown in Table 4 , all the lower bound of variant VAEs can be described as Reconstruction Loss+ Regularization where all the Regularization term and the hyper-parameters are given in this table.

<|TLDR|>

@highlight

disentangled representation learning