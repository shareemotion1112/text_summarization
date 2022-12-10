Generative models have been successfully applied to image style transfer and domain translation.

However, there is still a wide gap in the quality of results when learning such tasks on musical audio.

Furthermore, most translation models only enable one-to-one or one-to-many transfer by relying on separate encoders or decoders and complex, computationally-heavy models.

In this paper, we introduce the Modulated Variational auto-Encoders (MoVE) to perform musical timbre transfer.

First, we define timbre transfer as applying parts of the auditory properties of a musical instrument onto another.

We show that we can achieve and improve this task by conditioning existing domain translation techniques with Feature-wise Linear Modulation (FiLM).

Then, by replacing the usual adversarial translation criterion by a Maximum Mean Discrepancy (MMD) objective, we alleviate the need for an auxiliary pair of discriminative networks.

This allows a faster and more stable training, along with a controllable latent space encoder.

By further conditioning our system on several different instruments, we can generalize to many-to-many transfer within a single variational architecture able to perform multi-domain transfers.

Our models map inputs to 3-dimensional representations, successfully translating timbre from one instrument to another and supporting sound synthesis on a reduced set of control parameters.

We evaluate our method in reconstruction and generation tasks while analyzing the auditory descriptor distributions across transferred domains.

We show that this architecture incorporates generative controls in multi-domain transfer, yet remaining rather light, fast to train and effective on small datasets.

<|TLDR|>

@highlight

The paper uses Variational Auto-Encoding and network conditioning for Musical Timbre Transfer, we develop and generalize our architecture for many-to-many instrument transfers together with visualizations and evaluations.

@highlight

Proposes a Modulated Variational auto-Encoder to perform musical timbre transfer by replacing the usual adversarial translation criterion by a Maxiimum Mean Discrepancy

@highlight

Describes a many-to-many model for musical timbre transfer which builds on recent developments in domain and style transfer

@highlight

Proposes a hybrid VAE-based model to perform timbre transfer on recordings of musical instruments.