In this paper, we propose an arbitrarily-conditioned data imputation framework built upon variational autoencoders and normalizing flows.

The proposed model is capable of mapping any partial data to a multi-modal latent variational distribution.

Sampling from such a distribution leads to stochastic imputation.

Preliminary evaluation on MNIST dataset shows promising stochastic imputation conditioned on partial images as input.

Neural network based algorithms have been shown effective and promising for various downstream tasks including classification (Deng et al., 2009; Damianou and Lawrence, 2013) , retrieval (Carvalho et al., 2018) , prediction (He et al., 2018) , and more.

In order to correctly learn how to perform these tasks, they usually rely strictly on access to fully-observed data.

However, acquiring this type of data in real life requires tremendous human effort, limiting the applicability of this family of models.

Having a framework designed to perform inference on partially-observed data will not only alleviate the aforementioned constraint, but also open possibilities to perform data imputation, in which the missing data is inferred.

Data imputation, also referred to conditional generation, has been an active research area (Little and Rubin, 1986; Song et al., 2018; Zadeh et al., 2019) .

The probabilistic nature of this task makes it difficult to adopt off-the-shelf deterministic models widely studied.

In other words, conditioned on the same partially-observed data as input, multiple plausible fully-observed data should be able to be imputed.

Variational autoencoders (VAEs) (Kingma and Welling, 2013) , as a popular probabilistic modelling approach, have been applied to the data imputation task recently.

A variational autoencoder defines a generative process that jointly models the distribution p θ (x, z) of the observed variable x and latent variable z, governed by parameters θ.

Instead of performing local inference, VAEs include an inference network parameterized by φ to output an approximate posterior distribution q φ (z|x).

Both the generative model and the inference model are optimized with a unified evidence lower bound (ELBO) on marginal data likelihood:

.

Recent literature on utilizing VAEbased models mainly focus on the effectiveness of combination of various obversed parts (Ma et al., 2019; Ivanov et al., 2018) .

Different from the related works described above, we propose to enrich the latent space of variational autoencoders to enable multi-modal posterior inference, and therefore probabilistic imputation.

Specifically, we use a two-stage model, with first-stage focusing on learning a representation space based on fully-observed data, and second-stage focusing on aligning the representation space embedded from partially-observed data to the one in stage-one.

Using flow-based transformations for constructing a rich latent distribution, the proposed model is capable of inferring multi-modal variational latent distributions.

Adopting a standard VAE approach for this problem would involve advocating for a model which receives partial data as input and, with the feedback of a standard reconstruction loss, learns to output the full data.

Training such a model would pose many challenges.

Firstly, gradient coming from the very end of the network would promote stronger imputation on the decoder, whereas the encoder could learn to simply encode the partial data.

Secondly, there would be no mechanism to ensure the distribution of possible reconstructions would be correctly captured by the proposed posterior, which is generated by the encoder and fully conditioned on the partial data.

To amortize these problems, we propose a two-stage schema, represented in Figure 1 .

The first stage (upper part of the figure) corresponds to the encoder of a VAE model.

This encoder was trained with an associated decoder, which was later discarded, with the task of encoding and reconstructing the full data.

If properly trained, this stage's proposed posterior correctly depicts a good distribution of the full data, because this is a requirement in order to also reconstruct it.

Once trained, its weights are fixed, and then the model of the second stage is trained on the partial data.

Note that the encoders and decoders of the first and second stages are different -they can have the same architecture but do not share weights.

Because the latent space of the first model is rich enough to represent the full data's distribution (under the perspective of the first model), we propose to adopt a divergence loss between the first and the second model.

This divergence acts as a distillation method, allowing the first model to inject rich information about the latent representation of the full data into the second-stage model.

This injection will ensure weak alignment between both representation spaces, while also providing direct feedback to the encoder about the expected distribution of data in that space.

One problem with using simple families of posterior approximation is the lack of support for modeling multi-modal distributions, in which a reconstruction can take multiple forms.

To compensate for that, we adopt a Normalizing Flow (Rezende and Mohamed, 2015) model inside the latent space, forcing the divergence between stage-one and stage-two to happen between the normal distribution, from the proposed posterior of the former, and the more complex distribution created by the flow model of the latter.

The nature of this divergence then becomes a problem: (1) How can we model a divergence between a simple and a more complex distribution for which we don't know the parameters?

(2) Once defined, how can we ensure a multi-modal distribution can be modeled by the second stage?

q(x i ) .

From this perspective, we can derive a Monte-Carlo approach for the KLDivergence, as long as p(x i ) and q(x i ) are tractable:

In our model, we know p(x i ) is coming from a normal distribution, which is the proposed posterior of stage-one, therefore we only have to address the computation of q(x i ), which is coming from the flow model.

Thanks to properties of Normalizing Flows (NFs), this can be modeled as a correction term applied to the simple distribution before the flow:

where K is the number of transformations f k , and q 0 (z 0 ) is the simple distribution that is transformed to the complex distribution q K (z K ) through flow transformations.

To complete the model we also added a second divergence loss between the simple distribution (prior to the NF) and a Gaussian centered at zero with standard deviation of one.

This extra divergence allows us to control the support of that distribution, regaining generative capability in all subsequent spaces, including the more complex one created by the NF module.

The second stage model (encoder, partial posterior and decoder) is trained from scratch with the reconstruction and the divergence losses.

During the training of the second stage model, the first stage model is fixed and it provides supervision for the structure of the latent space.

Finally, problem (2) becomes irrelevant when we take into consideration the stochastic optimization in neural networks.

If the training data is rich enough to correctly represent the multi-modal nature of the full data (this is a base assumption for any machine learning model), the best way to minimize the divergence loss is, indeed, by creating a multi-modal distribution which has density directly proportional to the likelihood of the data.

In Figure 2 we present preliminary results which showcase the benefit of having each of the proposed modules.

For this experiment, a regular grid is defined inside the latent space, and values in this grid are sampled from the decoder to observe the latent structure organization.

In Figure 2 (a), we display results for a baseline approach, representing the best possible scenario, in which the encoder has access to the full data.

We then show, in Figure 2(b) , the same space when adopting the schema in Figure 1 , but without the NF module; and the full architecture -with NF -in Figure 2 (c).

We observe that the NF module allowed the network to have a more flexible latent space, when compared to the case without NF.

Following this experiment, we set out to test whether the multi-modality of reconstructions was being captured by the model.

Due to limited space, we limit ourselves to a single example, for which we don't penalize the model for not perfectly reconstructing the partial data -the goal is to verify if the multi-modality is being captured, and if the model is able to recognize the digit.

Figure 3(a) demonstrates the problem we're aiming to solve: given a partially observed piece of data, we want to capture all possible interpretations and reconstructions of the full data.

The results without NF and with NF are given in Figure 3 (b) and Figure 3(c) , respectively.

We observe that adding the flow module allows the model to more precisely represent the possible reconstructions of partial data.

While Figure 3 (b) still displays signs of averaging and confusion, most of the digits in Figure 3 (c) are clearly identifiable, and the multi-modality of the possible reconstructions is correctly depicted.

Figure 3(b) , for example, was unable to provide the possibility of "0" being a valid reconstruction to the partial provided in Figure 3(a) .

Although we demonstrate the power of our model in the simple case of MNIST, our model remains data-agnostic, and can be applied to any data modality (images, videos, text, sound, etc) .

Possible applications range from arbitrarily-conditioned data imputation to data generation following complex modality interactions, which are partly modeled by the NF inside the latent space.

<|TLDR|>

@highlight

We propose an arbitrarily-conditioned data imputation framework built upon variational autoencoders and normalizing flows