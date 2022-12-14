This paper is concerned with the robustness of VAEs to adversarial attacks.

We highlight that conventional VAEs are brittle under attack but that methods recently introduced for disentanglement such as β-TCVAE (Chen et al., 2018) improve robustness, as demonstrated through a variety of previously proposed adversarial attacks (Tabacof et al. (2016); Gondim-Ribeiro et al. (2018); Kos et al.(2018)).

This motivated us to develop Seatbelt-VAE, a new hierarchical disentangled VAE that is designed to be significantly more robust to adversarial attacks than existing approaches, while retaining high quality reconstructions.

Unsupervised learning of disentangled latent variables in generative models remains an open research problem, as is an exact mathematical definition of disentangling (Higgins et al., 2018) .

Intuitively, a disentangled generative model has a one-to-one correspondence between each input dimension of the generator and some interpretable aspect of the data generated.

For VAE-derived models (Kingma & Welling, 2013; Rezende et al., 2014) this is often based around rewarding independence between latent variables.

Factor VAE (Kim & Mnih, 2018) , β-TCVAE (Chen et al., 2018) and HFVAE (Esmaeili et al., 2019) have shown that the evidence lower bound can be decomposed to obtain a term capturing the degree of independence between latent variables of the model, the total correlation.

By up-weighting this term, we can obtain better disentangled representations under various metrics compared to β-VAEs (Higgins et al., 2017a) .

Disentangled representations, much like PCA or factor analysis, are not only human-interpretable but also offer more informative and robust latent space representations.

In addition, information theoretic interpretations of deep learning show that having a disentangled hidden layer within a discriminative deep learning model increases robustness to adversarial attack (Alemi et al., 2017) .

Adversarial attacks on deep generative models, more difficult than those on discriminative models (Tabacof et al., 2016; Gondim-Ribeiro et al., 2018; Kos et al., 2018) , attempt to fool a model into reconstructing a chosen target image by adding distortions to the original input image.

Generally, the most effective attack mode involves making the latent-space representation of the distorted input match that of the target image (Gondim-Ribeiro et al., 2018; Kos et al., 2018) .

This kind of attack is particularly relevant to applications where the encoder's output is used downstream.

Projections of data from VAEs, disentangled or not, are used for tasks such as: text classification (Xu et al., 2017) ; discrete optimisation (Kusner et al., 2017) ; image compression (Theis et al., 2017; Townsend et al., 2019) ; and as the perceptual part of a reinforcement learning algorithm (Ha & Schmidhuber, 2018; Higgins et al., 2017b) , the latter of which uses a disentangled VAE's encoder to improve the robustness of the agent to domain shift.

Here we demonstrate that β-TCVAEs are significantly more robust to 'latent-space' attack than standard VAEs, and are generally more robust to attacks that act to maximise the evidence lower bound for the adversarial input.

The robustness of these disentangled models is highly relevant because of the use-cases for VAEs highlighted above.

However, imposing additional disentangling constraints on a VAE training objective degrades the quality of resulting drawn or reconstructed images (Higgins et al., 2017a; Chen et al., 2018) .

We sought whether more powerful, expressive models, can help ameliorate this and in doing so built Figure 1: Latent-space adversarial attacks on Chairs, 3D Faces and CelebA for different models, including our proposed Seatbelt-VAE.

β = 10 for β-TCVAE, β-TCDLGM and Seatbelt-VAE.

L is the number of stochastic layers.

Clockwise within each plot we show the initial input, its reconstruction, the adversarial input, the adversarial distortion added to make it (shown normalised), the adversarial input's reconstruction, and the target image.

Following Tabacof et al. (2016) ; Gondim-Ribeiro et al. (2018) we attack with different degrees of penalisation on the magnitude of the adversarial distortion; in choosing the distortion to show, we pick the one with the penalisation that resulted in the value of the attack objective just better than the mean.

See Section 5 for more details.

a hierarchical disentangled VAE, Seatbelt-VAE, drawing on works like Ladder VAEs (Sønderby et al., 2016) and BIVA (Maaløe et al., 2019) .

We demonstrate that Seatbelt-VAEs are more robust to adversarial attacks than β-TCVAEs and β-TCDLGMs (the latter a simple generalisation we make of β-TC penalisation to hierarchical VAEs).

See Figure 1 for a demonstration.

Rather than being concerned with human-interpretable controlled generation by our models, which has been the focus of much research into disentangling, instead we are interested in the robustness afforded by disentangled representations.

Thus our key contributions are:

• A demonstration that β-TCVAEs are significantly more robust to adversarial attacks via their latents than vanilla VAEs.

• The introduction of Seatbelt-VAE, a hierarchical version of the β-TCVAE, designed to further increase robustness to various types of adversarial attack, while also giving better perceptual quality of reconstructions even when regularised.

Variational autoencoders (VAEs) are a deep extension of factor analysis suitable for high-dimensional data like images (Kingma & Welling, 2013; Rezende et al., 2014) .

They have a joint distribution over data x and latent variables z: p θ (x, z) = p θ (x|z)p(z) where p(z) = N (0, I) and p θ (x|z) is an appropriate distribution given the form of the data, the parameters of which are represented by deep nets with parameters θ.

As exact inference is intractable for this model, in a VAE we perform amortised stochastic variational inference.

By introducing an approximate posterior distribution q φ (z|x) = N (µ φ (x), Σ φ (x)), we can perform gradient ascent on the evidence lower bound (ELBO) L(x) = −D KL (q φ (z|x)||p θ (x, z)) = E q φ (z|x) log p θ (x|z) − D KL (q φ (z|x)||p(z)) ≥ log p(x) w.r.t.both θ and φ jointly, using the reparameterisation trick to take gradients through Monte Carlo samples from q φ (z|x).

In a β-VAE (Higgins et al., 2017a) , a free parameter β multiplies the D KL term in L(x) above.

This objective L β (x) remains a lower bound on the evidence.

Decompositions of L(x) shed light on its meaning.

As shown in Hoffman & Johnson (2016) ; Makhzani et al. (2016) ; Kim & Mnih (2018) ; Chen et al. (2018) ; Esmaeili et al. (2019) , one can define the evidence lower bound not per data-point, but instead write it over a dataset D of size N , D = {x n }, so we have L(θ, φ, D).

Esmaeili et al. (2019) gives a decomposition of this dataset-level evidence lower bound:

= E q φ (z,x) log p θ (x|z)

where under the assumption that p(z) factorises we can further decompose 4 :

where j indexes over coordinates in z. q φ (z, x) = q φ (z|x)q(x) and q(x) :

is the empirical data distribution.

q φ (z) := 1 N N n=1 q φ (z|x n ) is called the average encoding distribution following Hoffman & Johnson (2016) .

A is the total correlation (TC) for q φ (z), a generalisation of mutual information to multiple variables (Watanabe, 1960) .

With this mean-field p(z), Factor and β-TCVAEs upweight this term, so we have an objective: L βTC (θ, φ, D) = 1 + 2 + 3 + B + β A (4) Chen et al. (2018) gives a differentiable, stochastic approximation to E q φ (z) log q φ (z), rendering this decomposition simple to use as a training objective using stochastic gradient descent.

We also note that A , the total correlation, is also the objective in Independent Component Analysis (Bell & Sejnowski, 1995; Roberts & Everson, 2001 ).

We now have a set of L layers of z variables: z = [z 1 , z 2 , ..., z L ].

The evidence lower bound for models of this form is:

(5) The simplest VAE with a hierarchy of conditional stochastic variables in the generative model is the Deep Latent Gaussian Model (DLGM) of Rezende et al. (2014) .

The forward model factorises as a chain:

Each p θ (z i |z i+1 ) is a Gaussian distribution with mean and variance parameterised by deep nets.

p(z L ) is a unit isotropic Gaussian.

We can understand this additional expressive power as coming from having a richer family of distributions for the likelihood over data x marginalising out all intermediate layers:

is a non-Gaussian, highly flexible, distribution.

To perform amortised variational inference one introduces a recognition network, which can be any directed acyclic graph where each node, each distribution over each z i , is Gaussian conditioned on its parents.

This could be a chain, as in Rezende et al. (2014) :

Again, marginalising out intermediate

is a non-Gaussian, highly flexible, distribution.

However, training DLGMs is challenging: the latent variables furthest from the data can fail to learn anything informative (Sønderby et al., 2016; Zhao et al., 2017) .

Due to the factorisation of q φ (z|x) and p θ (x, z) in a DLGM, it is possible for a single-layer VAE to train in isolation within a hierarchical model: each p θ (z i |z i+1 ) distribution can become a fixed distribution not depending on z i+1 such that each D KL divergence present in the objective between corresponding z i layers can still be driven to a local minima.

Zhao et al. (2017) gives a proof of this separation for the case where the model is perfectly trained, i.e.

This is the hierarchical version of the collapse of z units in a single-layer VAE (Burda et al., 2016) , but now the collapse is over entire layers z i .

It is part of the motivation for the Ladder VAE (Sønderby et al., 2016) and BIVA (Maaløe et al., 2019) .

We propose novel hierarchical disentangled VAEs where we aim to disentangle only in the top-most latent variables z L .

Following the Factor and β-TCVAEs we upweight the term of the form of A for z L .

Empirically we find models of this type are unable to converge when disentangling at the bottom most layer, or when disentangling at each layer.

Intuitively, we want to capture high-level disentangled information at the top, but leave lower layers free to learn rich entangled representations.

If p θ (x|z) = p θ (x|z 1 ), we obtain the generalisation of β-TC penalisation to a DLGM and call it β-TCDLGM.

It suffers from the problems of collapse described above.

Inspired by BIVA (Maaløe et al., 2019) , we choose instead to condition our likelihood on all z i layers:

Combining Eqs (7, 5, 8) and applying β-TC penalisation to the D KL term over z L :

where j is indexing over the coordinates in z L .

See Appendix for the derivation.

We call this model Seatbelt-VAE, as with the extra conditional dependencies and nodes we increase the safety of our model to adversarial attacks, to noise, and to decreases in perceptual quality as β increases.

We find that using free-bits regularisation (Kingma et al., 2016) greatly ameliorates the optimisation challenges associated with DLGMs.

For L = 1 this reduces to a β-TCVAE, and for L > 1, β = 1 it produces a DLGM with our augmented likelihood function.

For completeness, note that for β-TCDLGM:

VAEs and derived models are commonly trained using stochastic gradient ascent on the ELBO, on minibatches of the training data.

With the ELBO in Eq (9), this would be challenging because of the presence of average encoding distributions, which depend on the entire dataset.

To avoid having to handle large mixture distributions in our objective functions, we derive minibatch estimators that are a simple generalisation to disentangled hierarchical VAEs of the Minibatch Weighted Sampling estimator proposed in Chen et al. (2018) in the context of β-TCVAEs.

See Appendix for further details.

Most adversarial attack research has focused on discriminative models (Akhtar & Mian, 2018; Gilmer et al., 2018) and recently VAEs have found use in protecting discriminative models against attack (Schott et al., 2019; Ghosh et al., 2019) .

Currently, two adversarial modes have been proposed for attacking VAEs (Tabacof et al., 2016; Gondim-Ribeiro et al., 2018; Kos et al., 2018) .

In both attack modes the adversary wants draws from the model to be close to a target image x t , when given a distorted image x * = x + d as input.

When attacking a discriminative model the aim is to manipulate the comparatively low-dimensional output layer of the network, commonly aiming with the attack to diminish or increase only a handful of the output units.

However, for a generative model, the attacker is aiming to change a large number of pixel values in the output, changing the content of the reconstruction.

Intuitively this is a harder task, and the attacks proposed in the above papers do not always result in adversarial examples that are very close to the initial image in appearance.

The first mode of attack, which we call the output attack, aims to reward draws from the decoder conditioned on z ∼ q φ (z|x * ) that are close to x t via the ELBO.

For a vanilla VAE, this attack's adversarial objective is:

The second mode of attack, the latent attack, aims to find x * = x + d such that q φ (z|x * ) ≈ q φ (z|x t ) under some similarity measure r(·, ·), which implicitly means that the likelihood p θ (x t |z) is high when conditioned on draws from the posterior of the adversarial example.

This attack is important if one is concerned with using the encoder network of a VAE as part of downstream task.

For a single stochastic layer VAE, the latent-space adversarial objective is:

Note that both modes of attack penalise the L 2 norm of d, prioritising smaller distortions.

We denote samples from q φ (z|x + d) asz.

For Tabacof et al. (2016) ; Gondim-Ribeiro et al. (2018)

)

and for Kos et al. (2018) it is the L 2 distance ||z − z * || 2 ,z ∼ q φ (z|x + d), z * ∼ q φ (z|x) between draws from the corresponding posteriors or ||µ φ (x) − µ φ (x + d)|| 2 between their means.

We follow the former papers and use the D KL formulation.

All three papers find that the latent attack mode is as or more effective than the output attack for single layer VAEs both under perceptual evaluation and various proposed metrics (Tabacof et al., 2016; Gondim-Ribeiro et al., 2018; Kos et al., 2018) .

For latent attacks, the choice of which layers to attack depends on model architecture.

For DLGMs and β-TCDLGMs the attacker only needs to match at the bottom latent layer as p θ (x|z) = p θ (x|z 1 ), see Eq (7).

See Appendix for plots showing how effective this attack is regardless of β and L.

Even though the decoder is conditioned on all latent layers, one could choose to attack individual layers for Seatbelt-VAE.

For example, one could attack just the first layer z 1 .

If one were able to find a perfect latent-space attack in

, then the variational posteriors in higher layers would also be well matched.

Attacks that do not perfectly match the target z 1 may have their mismatch with the target posterior amplified in higher layers.

In Seatbelt-VAE the likelihood over data is conditioned on all z layers, being off-target in these higher layers matters.

In the Appendix we show that targeting the top or base layers individually is not as effective as attacking all layers.

Hence:

Here we perform four tranches of experiments.

Firstly, we demonstrate that the reconstructions given by Seatbelt-VAEs (and β-TCDLGMs) degrade much less strongly as β is increased than in β-TCVAEs.

Secondly, we perform a variety of adversarial attacks on all models.

We demonstrate that increasing β makes β-TCVAEs more robust to adversarial attacks than vanilla VAEs, and that Seatbelt-VAEs are more robust still.

Thirdly, we show that these disentangled models are most robust than vanilla VAEs to unstructured noise distorting their inputs, with Seatbelt-VAEs again the most robust.

Finally, we study the effect of disentangling on the sparsity of model weights.

We perform these experiments on Chairs (Aubry et al., 2014) , 3D faces (Paysan et al., 2009) , and CelebA (Liu et al., 2015) .

Additional results for dSprites (Higgins et al., 2017a) can be found in the Appendix.

We used the same encoder and decoder architectures as Chen et al. (2018) for each dataset.

For the details of neural network architectures and training, see Appendix and accompanying code.

To show the degree to which our models are disentangling, the Appendix also contains the Mutual Information Gap (MIG) (Chen et al., 2018) at the top layer of each model.

Though our models obtain high MIG at z L , this does not imply that decoding from latent traversals in z L will result in the generation of images with human-interpretable factors of variation.

This is made abundantly clear in the latent space traversal plots, also shown in the Appendix.

As such, we do not believe existing disentangling metrics directly apply to hierarchical models.

We trained β-TCVAEs, β-TCDLGMs, and Seatbelt-VAEs for a range of β penalisations.

In Figure 3 we plot the final ELBO of our trained models, but calculated without the additional β penalisation that was applied during training.

The ELBO for β-TCVAE [Eq (4)] declines with β much more quickly than Seatbelt VAEs [Eq (10)] or β-TCDLGMs [Eq (11)].

In the Appendix we also show that increasing β reduces D KL collapse.

This is interesting, as it shows that we can increase the β penalisation for Seatbelt-VAEs, without a large degradation in the quality of the model as measured by the ELBO.

In Figure 4 we see the effect of depth and disentangling on reconstructions of CelebA. The bottom row, showing the reconstructions from a Seatbelt-VAE with L = 4 and β = 20 clearly maintains facial identity better than those from a β-TCVAE in the middle row.

The effect is clearest for the 3 rd , 4 th and 7 th columns, where many of the individuals' finer facial features are lost by the β-TCVAE but maintained by the Seatbelt-VAE.

This fits with the results in Figure 3 , and shows that resistance of the quality of the reconstructions of Seatbelt to increasing β is visually perceptible as well as measurable. (11) and (10) respectively].

Shading corresponds to the 95% CI over variation due to variation of ||z|| and L. Shading corresponds to the 95% CI over variation due to our stable of images and our values of ||z|| and λ.

We apply attacks minimising each of ∆ output and ∆ latent on: vanilla VAEs, β-TCVAEs, β-TCDLGMs and Seatbelt-VAEs; trained on: Chairs (Aubry et al., 2014) , 3D faces (Paysan et al., 2009) , and CelebA (Liu et al., 2015) ; for a range of β, L and λ values.

We randomly sampled 10 input-target pairs for each dataset.

We prefer to avoid classifier based metrics (Kos et al., 2018) as in general we think that such analysis can be hard to interpret given the many available choices of classifier.

Instead, we evaluate the effectiveness of adversarial attacks from the values reached by − log p θ (x t |z), by the attack objectives {∆ output , ∆ latent } and by visually appraising the adversarial input (x + d) and the adversarial reconstruction.

Note that higher values of − log p θ (x t |z), ∆ output , ∆ latent indicate less effective attacks.

Figure 1 shows latent space attacks and demonstrates that they are less effective on disentangled models.

As in Gondim-Ribeiro et al. (2018) , we are showing the attack for the λ that gives us an attack objective just better than the average objective over all attacks tried.

Note that for Seatbelt-VAEs, for high values of β and L latent attacks often result in the outputs from adversarial attack resembling the original inputs.

See Appendix for more examples of the attacks for {∆ latent , ∆ output } for the models trained on dSprites (a toy dataset for disentangling), Chairs, 3D Faces and CelebA; each over a range of values for β, L, and λ.

Note that we rarely observe perceptually effective output attacks regardless of model or settings, though vanilla VAEs are the most susceptible.

One might expect that adversarial attacks targeting a single factor of the data would be easier for the attacker.

However, we find that disentangled models protect effectively against these attacks as well.

See the Appendix for plots showing an attacker attempting to rotate a dSprites heart.

Figure 5 quantitatively shows that β-TCVAEs become harder to attack as β increases.

The values of ∆ latent for β-TCVAEs are ≈ 10 3 times higher than for a standard VAE on Chairs, and still greater than a factor of 10 for 3D faces.

∆ output attack is also less effective, by a smaller factor ≈ 1.2.

Figure 6 shows − log p θ (x t |z latent/output ) and Figure 7 shows ∆ latent/output over a range of datasets for β-TCDLGMs and Seatbelt-VAEs, varying L and β.

Larger values of these metrics correspond to less successful adversarial attacks.

Generally, β-TCDLGMs are very sensitive to latent attack, as we expect.

Like β-TCVAEs, Seatbelt-VAEs offer significant protection to latent attacks, and somewhat increased protection to output attacks compared to vanilla VAEs.

For Seatbelt-VAEs, as we go to the largest values of β and L for both Chairs and 3D Faces, ∆ latent grows by a factor of ≈ 10 7 .

The bottom rows of Figures 6 & 7 (c) (d) have L = 1, and thus correspond to β-TCVAEs.

They contain relatively low values of the adversarial objectives compared to L > 1.

Similarly the first column, corresponding to β=1 models, contains relatively low values.

These results tell us that depth and disentangling together offer the most effective protection from the adversarial attacks studied.

In the Appendix we also calculate the L 2 distance between target images and adversarial outputs and show that the loss of effectiveness of adversarial attacks is not due to the degradation of reconstruction quality from increasing β.

By these metrics too Seatbelt-VAEs outperform other models.

In addition to studying the robustness of these models to highly structured distortion, we can also consider robustness to random noise.

We add ∼ N (0, I) to the datasets, which are scaled to −1 ≤ x ≤ 1, and then evaluate E q φ (z|x+ ) p θ (x|z * ), where z * corresponds to the encoder embedding of x + and x is the original (non-noisy) data.

See Figure 8 for smoothed histogram plots of this for different models for different degrees of β.

Both β-TC and Seatbelt-VAEs are effectively denoising autoencoders.

They become more robust to noise with increasing β, while β-TCDLGMs get worse.

See Appendix for plots showing the robustness of these models to smaller magnitude noise.

Some of the robustness of disentangled models to adversarial attacks may be conferred by their robustness to random perturbations of their inputs.

In the auto-encoder view of these models, the D KL terms in L(θ, φ, D) are associated with a form of regularisation of the model (Doersch, 2016) .

Recent work shows that for linear autoencoders, L 2 regularisation of the weights corresponds to orthogonality of the latent projections (Kunin et al., 2019) .

For deep models we expect that disentangling is associated with regularised decoders and more complex encoders.

The decoder receives a simpler representation, but building this representation requires more calculation.

Here we measure the L 2 norm of the weights of our networks as a function of β, shown in Table 1 .

See Appendix for results for β-TCDLGM.

As we increase β for β-TCVAEs and Seatbelt-VAEs for Chairs, 3D Faces, and CelebA the L 2 norm increases for the encoder and decreases for the decoder.

A more complex encoder is more difficult to match in the latent space and regularised decoders may be contributing to the denoising properties seen in Figure 8 .

That the changes are generally greater for β-TCVAE than Seatbelt-VAE makes sense, as the encoder and decoder of the former interact directly with the disentangled representation.

For the latter the decoder receives inputs from all z i , of varying degrees of disentanglement.

We have presented the increases in robustness to adversarial attack afforded by β-TCVAEs.

This increase in robustness is strongest for attacks via the latent space.

While disentangled models are often motivated by their ability to provide interpretable conditional generation, many use cases for VAEs centre on the learnt latent representation of data.

Given the use of these representations as inputs for other tasks, the latent attack mode is the most important to protect against.

Recent work by Shamir et al. (2019) gives a constructive proof for the existence of adversarial inputs for deep neural network classifiers with small Hamming distances.

The proof holds with deterministic defence procedures that work as additional deterministic layers of the networks, and in the presence of adversarial training (Szegedy et al., 2014; Ganin et al., 2016; Tramèr et al., 2018; Shaham et al., 2018) .

Shamir et al. (2019) thus give a theoretical grounding for using stochastic methods to defend against adversarial inputs.

As VAEs are already used to defend deep net classifiers (Schott et al., 2019; Ghosh et al., 2019) , more robust VAEs, like β-TCVAEs, could find use in this area.

We introduce Seatbelt-VAE, a particular hierarchical VAE disentangled on the top-most layer with skip connections down to the decoder.

This model further increases robustness to adversarial attacks, while also increasing the quality of reconstructions.

The performance of our model under adversarial attack to robustness is mirrored in robustness to uncorrelated noise: these models are effective denoising autoencoders as well.

We hope this work stimulates further interest in defending and attacking VAEs.

(5) cf.

Eq (7) in the main paper.

The likelihood is conditioned on all z layers:

Now we have:

Apply βTC decomposition to T as in Chen et al. (2018) .

j indexes over units in z L .

(A.14)

for our chosen generative model.

As in Chen et al. (2018), we choose to weight T b , the total correlation for q φ (z L ), by a prefactor β.

Giving us the ELBO for Seatbelt-VAEs, Eq (10).

As in Chen et al. (2018) , applying β-TC decomposition requires us to calculate terms of the form:

The i = 1 case is covered in the appendix of Chen et al. (2018) .

First we will repeat the argument for i = 1 as made in Chen et al. (2018) , but in our notation, and then we cover the case i > 1 for models with factorisation of q φ (z|x) as in Eq 7 in the main paper.

, the probability of a sampled minibatch given that one member is x and the remaining M − 1 points are sampled iid from q(x), so r(

So then during training, one samples a minibatch {x 1 , x 2 , ..., x M } and can estimate E q φ (z 1 ) log q φ (z 1 ) as:

and z 1 i is a sample from q φ (z 1 |x i ).

.

Now instead of having a minibatch of datapoints, we have a minibatch of draws of

Each member of which is the result of sequentially sampling along a chain, starting with some particular datapoint x m ∼ q(x).

Thus each member of this batch B i−1 M is the descendant of a particular datapoint that was sampled in an iid minibatch B M as defined above.

We similarly define r(B i−1 M |z i−1 ) as the probability of selecting a particular minibatch B i−1 M of these values out from our set {z i−1 n } (of cardinality N ) given that we have selected into our minibatch one particular z i−1 from these N values.

Like above,

Where we have followed the same steps as in the previous subsection.

During training, one samples a minibatch {z

M }, where each is constructed by sampling ancestrally.

Then one can estimate E q φ (z i ) log q φ (z i ) as:

and z i k is a sample from q φ (z i |z i−1 k ).

In our model we only need terms of this form for i = L, so we have:

All runs were done on the Azure cloud system on NC6 GPU machines.

For β-TCDLGMs and Seatbelt-VAEs we also have the mappings q φ (z i+1 |z i ) and p θ (z i |z i+1 ).

These are amortised as MLPs with 2 hidden layers with batchnorm and Leaky-ReLU activation.

The dimensionality of the hidden layers also decreases as a function of layer index i: 1024, 512, 256, 128, 64] (C.8)

To train the model we used ADAM (Kingma & Ba, 2015) with default parameters and a learning rate of 0.001.

All data was preprocessed to fall on the interval -1 to 1.

CelebA and Chairs were both downsampled and cropped as in (Chen et al., 2018) and (Kulkarni et al., 2015) respectively.

(d),(g) the L 2 distance between x t and its reconstruction when given as input and the same between the adversarial input x * and its reconstruction; (b),(e),(h) the adversarial objectives ∆ output/image ; (c),(f),(i) − log p θ (x t |z), z ∼ q φ (z|x * ) and the MIG.

For a DLGM (Rezende et al., 2014) with 2-5 z layers, with q φ (z|x) factorised as in Eq (7), p θ (x, z) factorised as in Eq (6), and βTC penalisation applied to the top layer, we find that latent attacks targeted at z 1 are highly effective and remain so as L and β each increase.

These models are, however, slightly more robust to output attacks and this attack becomes less effective as β increases, but more effective as L increases.

The ease of attacking via z 1 is consistent with its separation out from the rest of the model.

Note that there is a reduction in diversity of the samples for L = 1 (ie a β-TC VAE), β = 10, which is not the case for the samples from the β = 10 L = 4 Seatbelt-VAE.

The Mutual Information Gap (Chen et al., 2018 ) is average over ground truth factors of variation of the entropy-normalised difference between the greatest mutual information between the any of the units in z and a given ground-truth factor of variation ν and the second-greatest such mutual information:

where z * j = arg max j I(z j , ν k ).

Note that MIG decreases as we increase |z|, indicating that we get degenerate latent representations -that is different units in z end up with similar mutual information to the same ground truth factors.

The red line in a) is at |z| = 6, the number of ground-truth factors of variation for dSprites.

<|TLDR|>

@highlight

We show that disentangled VAEs are more robust than vanilla VAEs to adversarial attacks that aim to trick them into decoding the adversarial input to a chosen target. We then develop an even more robust hierarchical disentangled VAE, Seatbelt-VAE.

@highlight

The authors propose a new VAE model called seatbelt-VAE, showing to be more robust for latent attack than benchmarks.