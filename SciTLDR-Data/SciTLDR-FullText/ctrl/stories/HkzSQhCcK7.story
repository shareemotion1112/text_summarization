Convolutional architectures have recently been shown to be competitive on many sequence modelling tasks when compared to the de-facto standard of recurrent neural networks (RNNs) while providing computational and modelling advantages due to inherent parallelism.

However, currently, there remains a performance gap to more expressive stochastic RNN variants, especially those with several layers of dependent random variables.

In this work, we propose stochastic temporal convolutional networks (STCNs), a novel architecture that combines the computational advantages of temporal convolutional networks (TCN) with the representational power and robustness of stochastic latent spaces.

In particular, we propose a hierarchy of stochastic latent variables that captures temporal dependencies at different time-scales.

The architecture is modular and flexible due to the decoupling of the deterministic and stochastic layers.

We show that the proposed architecture achieves state of the art log-likelihoods across several tasks.

Finally, the model is capable of predicting high-quality synthetic samples over a long-range temporal horizon in modelling of handwritten text.

Generative modeling of sequence data requires capturing long-term dependencies and learning of correlations between output variables at the same time-step.

Recurrent neural networks (RNNs) and its variants have been very successful in a vast number of problem domains which rely on sequential data.

Recent work in audio synthesis, language modeling and machine translation tasks BID8 BID9 BID13 has demonstrated that temporal convolutional networks (TCNs) can also achieve at least competitive performance without relying on recurrence, and hence reducing the computational cost for training.

Both RNNs and TCNs model the joint probability distribution over sequences by decomposing the distribution over discrete time-steps.

In other words, such models are trained to predict the next step, given all previous time-steps.

RNNs are able to model long-term dependencies by propagating information through their deterministic hidden state, acting as an internal memory.

In contrast, TCNs leverage large receptive fields by stacking many dilated convolutions, allowing them to model even longer time scales up to the entire sequence length.

It is noteworthy that there is no explicit temporal dependency between the model outputs and hence the computations can be performed in parallel.

The TCN architecture also introduces a temporal hierarchy: the upper layers have access to longer input sub-sequences and learn representations at a larger time scale.

The local information from the lower layers is propagated through the hierarchy by means of residual and skip connections BID2 .However, while TCN architectures have been shown to perform similar or better than standard recurrent architectures on particular tasks BID2 , there currently remains a performance gap to more recent stochastic RNN variants BID3 BID7 BID11 BID12 BID14 BID25 .

Following a similar approach to stochastic RNNs, BID21 present a significant improvement in the log-likelihood when a TCN model is coupled with latent variables, albeit at the cost of limited receptive field size.

The computational graph of generative (left) and inference (right) models of STCN.

The approximate posterior q is conditioned on dt and is updated by the prior p which is conditioned on the TCN representations of the previous time-step dt???1.

The random latent variables at the upper layers have access to a long history while lower layers receive inputs from more recent time steps.

In this work we propose a new approach for augmenting TCNs with random latent variables, that decouples deterministic and stochastic structures yet leverages the increased modeling capacity efficiently.

Motivated by the simplicity and computational advantages of TCNs and the robustness and performance of stochastic RNNs, we introduce stochastic temporal convolutional networks (STCN) by incorporating a hierarchy of stochastic latent variables into TCNs which enables learning of representations at many timescales.

However, due to the absence of an internal state in TCNs, introducing latent random variables analogously to stochastic RNNs is not feasible.

Furthermore, defining conditional random variables across time-steps would result in breaking the parallelism of TCNs and is hence undesirable.

In STCN the latent random variables are arranged in correspondence to the temporal hierarchy of the TCN blocks, effectively distributing them over the various timescales (see FIG0 .

Crucially, our hierarchical latent structure is designed to be a modular add-on for any temporal convolutional network architecture.

Separating the deterministic and stochastic layers allows us to build STCNs without requiring modifications to the base TCN architecture, and hence retains the scalability of TCNs with respect to the receptive field.

This conditioning of the latent random variables via different timescales is especially effective in the case of TCNs.

We show this experimentally by replacing the TCN layers with stacked LSTM cells, leading to reduced performance compared to STCN.We propose two different inference networks.

In the canonical configuration, samples from each latent variable are passed down from layer to layer and only one sample from the lowest layer is used to condition the prediction of the output.

In the second configuration, called STCN-dense, we take inspiration from recent CNN architectures BID18 and utilize samples from all latent random variables via concatenation before computing the final prediction.

Our contributions can thus be summarized as: 1) We present a modular and scalable approach to augment temporal convolutional network models with effective stochastic latent variables.

2) We empirically show that the STCN-dense design prevents the model from ignoring latent variables in the upper layers BID32 .

3) We achieve state-of-the-art log-likelihood performance, measured by ELBO, on the IAM-OnDB, Deepwriting, TIMIT and the Blizzard datasets.

4) Finally we show that the quality of the synthetic samples matches the significant quantitative improvements.

Auto-regressive models such as RNNs and TCNs factorize the joint probability of a variable-length sequence x = {x 1 , . . .

, x T } as a product of conditionals as follows: DISPLAYFORM0 where the joint distribution is parametrized by ??.

The prediction at each time-step is conditioned on all previous observations.

The observation model is frequently chosen to be a Gaussian or Gaussian mixture model (GMM) for real-valued data, and a categorical distribution for discrete-valued data.

In TCNs the joint probabilities in Eq.(1) are parametrized by a stack of convolutional layers.

Causal convolutions are the central building block of such models and are designed to be asymmetric such that the model has no access to future information.

In order to produce outputs of the same size as the input, zero-padding is applied at every layer.

In the absence of a state transition function, a large receptive field is crucial in capturing long-range dependencies.

To avoid the need for vast numbers of causal convolution layers, typically dilated convolutions are used.

Exponentially increasing the dilation factor results in an exponential growth of the receptive field size with depth BID31 BID2 .

In this work, without loss of generality, we use the building blocks of Wavenet as gated activation units have been reported to perform better.

A deterministic TCN representation d l t at time-step t and layer l summarizes the input sequence x 1:t : DISPLAYFORM0 where the filter width is 2 and j denotes the dilation step.

In our work, the stochastic variables z l , l = 1 . . .

L are conditioned on TCN representations d l that are constructed by stacking K Wavenet blocks over the previous d l???1 (for details see FIG6 in Appendix).

VAEs BID20 BID24 introduce a latent random variable z to learn the variations in the observed non-sequential data where the generation of the sample x is conditioned on the latent variable z. The joint probability distribution is defined as: DISPLAYFORM0 and parametrized by ??.

Optimizing the marginal likelihood is intractable due to the non-linear mappings between z and x and the integration over z. Instead the VAE framework introduces an approximate posterior q ?? (z|x) and optimizes a lower-bound on the marginal likelihood: DISPLAYFORM1 where KL denotes the Kullback-Leibler divergence.

Typically the prior p ?? (z) and the approximate q ?? (z|x) are chosen to be in simple parametric form, such as a Gaussian distribution with diagonal covariance, which allows for an analytical calculation of the KL-term in Eq. (4).

An RNN captures temporal dependencies by recursively processing each input, while updating an internal state h t at each time-step via its state-transition function: DISPLAYFORM0 where f (h) is a deterministic transition function such as LSTM BID17 or GRU BID6 cells.

The computation has to be sequential because h t depends on h t???1 .The VAE framework has been extended for sequential data, where a latent variable z t augments the RNN state h t at each sequence step.

The joint distribution p ?? (x, z) is modeled via an auto-regressive model which results in the following factorization: DISPLAYFORM1 In contrast to the fixed prior of VAEs, N (0, I), sequential variants define prior distributions conditioned on the RNN hidden state h and implicitly on the input sequence x BID7 . , and the inference model (right), which is shared by both variants.

Diamonds represent the outputs of deterministic dilated convolution blocks where the dependence of dt on the past inputs is not shown for clarity (see Eq. FORMULA1 ).

xt and zt are observable inputs and latent random variables, respectively.

The generative task is to predict the next step in the sequence, given all past steps.

Note that in the STCN-dense variant the next step is conditioned on all latent variables z l t for l = 1 . . .

L.

The mechanics of STCNs are related to those of VRNNs and LVAEs.

Intuitively, the RNN state h t is replaced by temporally independent TCN layers d l t .

In the absence of an internal state, we define hierarchical latent variables z l t that are conditioned vertically, i.e., in the same time-step, but independent horizontally, i.e., across time-steps.

We follow a similar approach to LVAEs in defining the hierarchy in a top-down fashion and in how we estimate the approximate posterior.

The inference network first computes the approximate likelihood, and then this estimate is corrected by the prior, resulting in the approximate posterior.

The TCN layers d are shared between the inference and generator networks, analogous to VRNNs BID7 .

To preserve the parallelism of TCNs, we do not introduce an explicit dependency between different time-steps.

However, we suggest that conditioning a latent variable z l???1 t on the preceding variable z l t implicitly introduces temporal dependencies.

Importantly, the random latent variables in the upper layer have access to a larger receptive field due to its deterministic input d l t???1 , whereas latent random variables in lower layers are updated with different, more local information.

However, the latent variable z l???1 t may receive longer-range information from z l t .

The generative and inference models are jointly trained by optimizing a step-wise variational lower bound on the log-likelihood BID20 BID24 .

In the following sections we describe these components and build up the lower-bound for a single time-step t.

Each sequence step x t is generated from a set of latent variables z t , split into layers as follows: DISPLAYFORM0 where DISPLAYFORM1 Here the prior is modeled by a Gaussian distribution with diagonal covariance, as is common in the VAE framework.

The subscript p denotes items of the generative distribution.

For the inference distribution we use the subscript q. The distributions are parameterized by a neural network f

We propose two variants of the observation model.

In the non-sequential scenario, the observations are defined to be conditioned on only the last latent variable in the hierarchy, i.e., p ?? (x t |z 1 t ), following S??nderby et al. FORMULA0 ; BID16 and BID24 our STCN variant uses the same observation model, allowing for an efficient optimization.

However, latent units are likely to become inactive during training in this configuration BID5 BID4 BID32 resulting in a loss of representational power.

The latent variables at different layers are conditioned on different contexts due to the inputs d l t .

Hence, the latent variables are expected to capture complementary aspects of the temporal context.

To propagate the information all the way to the final prediction and to ensure that gradients flow through all layers, we take inspiration from BID18 and directly condition the output probability on samples from all latent variables.

We call this variant of our architecture STCN-dense.

The final predictions are then computed by the respective observation functions: DISPLAYFORM0 where f (o) corresponds to the output layer constructed by stacking 1D convolutions or Wavenet blocks depending on the dataset.

In the original VAE framework the inference model is defined as a bottom-up process, where the latent variables are conditioned on the stochastic layer below.

Furthermore, the parameterization of the prior and approximate posterior distributions are computed separately BID5 BID24 .

In contrast, propose a top-down dependency structure shared across the generative and inference models.

From a probabilistic point of view, the approximate Gaussian likelihood, computed bottom-up by the inference model, is combined with the Gaussian prior, computed top-down from the generative model.

We follow a similar procedure in computing the approximate posterior.

First, the parameters of the approximate likelihood are computed for each stochastic layer l: DISPLAYFORM0 followed by the downward pass, recursively computing the prior and approximate posterior by precision-weighted addition: DISPLAYFORM1 Finally, the approximate posterior has the same decomposition as the prior (see Eq. FORMULA6 ): DISPLAYFORM2 DISPLAYFORM3 Note that the inference and generative network share the parameters of dilated convolutions Conv (l) .

The variational lower-bound on the log-likelihood at time-step t can be defined as follows: DISPLAYFORM0 Using the decompositions from Eq. FORMULA6 and FORMULA0 , the Kullback-Leibler divergence term becomes: DISPLAYFORM1 The KL term is the same for the STCN and STCN-dense variants.

The reconstruction term L Recon t , however, is different.

In STCN we only use samples from the lowest layer of the hierarchy, whereas in STCN-dense we use all latent samples in the observation model: DISPLAYFORM2 DISPLAYFORM3 In the dense variant, samples drawn from the latent variables z l t are carried over the dense connections.

Similar to , the expectation over z One alternative option to use the latent samples could be to sum individual samples before feeding them into the observation model, i.e., sum([z .

We empirically found that this does not work well in STCN-dense.

Instead, we concatenate all samples [z DISPLAYFORM4 DISPLAYFORM5 analogously to DenseNet BID18 and BID19 .

We evaluate the proposed variants STCN and STCN-dense both quantitatively and qualitatively on modeling of digital handwritten text and speech.

We compare with vanilla TCNs, RNNs, VRNNs and state-of-the art models on the corresponding tasks.

In our experiments we use two variants of the Wavenet model: (1) the original model proposed in and (2) a variant that we augment with skip connections analogously to STCN-dense.

This additional baseline evaluates the benefit of learning multi-scale representations in the deterministic setting.

Details of the experimental setup are provided in the Appendix.

Our code is available at https://ait.ethz.ch/projects/2019/stcn/.Handwritten text: The IAM-OnDB and Deepwriting datasets consist of digital handwriting sequences where each time-step contains real-valued (x, y) pen coordinates and a binary pen-up event.

The IAM-OnDB data is split and pre-processed as done in BID7 .

BID1 extend this dataset with additional samples and better pre-processing.

BID7 26643 7413 1358 528 * VRNN (Normal) BID7 ??? 30235 ??? 9516 ??? 1354 ??? 495 * VRNN (GMM) BID7 ??? 29604 ??? 9392 ??? 1384 ??? 673 * SRNN (Normal) BID12 ??? 60550 ??? 11991 n/a n/a Z-forcing (Normal) BID14 ??? 70469 ??? 15430 n/a n/a Var.

Bi-LSTM (Normal) BID25 ??? 73976 ??? 17319 n/a n/a SWaveNet (Normal) BID21 ??? 72463 ??? 15708 DISPLAYFORM0 ??? 77438 ??? 17670 n/a n/a the STCN-dense version.

The same relative ordering is maintained on the Deepwriting dataset, indicating that the proposed architecture is robust across datasets.

Fig. 3 compares generated handwriting samples.

While all models produce consistent style, our model generates more natural looking samples.

Note that the spacing between words is clearly visible and most of the letters are distinguishable.

Speech modeling: TIMIT and Blizzard are standard benchmark dataset in speech modeling.

The models are trained and tested on 200 dimensional real-valued amplitudes.

We apply the same pre-processing as BID7 .

For this task we introduce STCN-dense-large, with increased model capacity.

Here we use 512 instead of 256 convolution filters.

Note that the total number of model parameters is comparable to SWaveNet and other SOA models.

On TIMIT, STCN-dense TAB0 significantly outperforms the vanilla TCN and RNN, and stochastic models.

On the Blizzard dataset, our model is marginally better than the Variational Bi-LSTM.

Note that the inference models of SRNN BID12 , Z-forcing BID14 , and Variational Bi-LSTM BID25 receive future information by using backward RNN cells.

Similarly, SWaveNet BID21 applies causal convolutions in the backward direction.

Hence, the latent variable can be expected to model future dynamics of the sequence.

In contrast, our models have only access to information up to the current time-step.

These results indicate that the STCN variants perform very well on the speech modeling task.

Latent Space Analysis: BID32 observe that in hierarchical latent variable models the upper layers have a tendency to become inactive, indicated by a low KL loss BID10 .

TAB2 shows the KL loss per latent variable and the corresponding log-likelihood measured by ELBO in our models.

Across the datasets it can be observed that our models make use of many of the latent variables which may explain the strong performance across tasks in terms of log-likelihoods.

Note that STCN uses a standard hierarchical structure.

However, individual latent variables have different information context due to the corresponding TCN block's receptive field.

This observation suggests that the proposed combination of TCNs and stochastic variables is indeed effective.

Furthermore, in STCN we see a similar utilization pattern of the z variables across tasks, whereas STCN-dense may have more flexibility in modeling the temporal dependencies within the data due to its dense connections to the output layer.

Replacing TCN with RNN: To better understand potential symergies between dilated CNNs and the proposed latent variable hierarchy, we perform an ablation study, isolating the effect of TCNs and the latent space.

To this end the deterministic TCN blocks are replaced with LSTM cells by keeping the latent structure intact.

We dub this condition LadderRNN.

We use the TIMIT and IAM-OnDB datasets for evaluation.

TAB3 summarizes performance measured by the ELBO.The most direct translation of the the STCN architecture into an RNN counterpart has 25 stacked LSTM cells with 256 units each.

Similar to STCN, we use 5 stochastic layers (see Appendix 7.1).

Note that stacking this many LSTM cells is unusual and resulted in instabilities during training.

Hence, the performance is similar to vanilla RNNs.

The second LadderRNN configuration uses 5 stacked LSTM cells with 512 units and a one-to-one mapping with the stochastic layers.

On the TIMIT dataset, all LadderRNN configurations show a significant improvement.

We also observe a pattern of improvement with densely connected latent variables.

This experiments shows that the proposed modular latent variable design does allow for the usage of different building blocks.

Even when attached to LSTM cells, it boosts the log-likelihood performance (see 5x512-LadderRNN), in particular when used with dense connections.

However, the empirical results suggest that the densely connected latent hierarchy interacts particularly well with dilated CNNs.

We suggest this is due to the hierarchical nature on both sides of the architecture.

On both datasets STCN models achieved the best performance and significantly improve with dense connections.

This supports our contribution of a latent variable hierarchy, which models different aspects of information from the input time-series.

DISPLAYFORM1 5 RELATED WORK BID24 propose Deep Latent Gaussian Models (DLGM) and propose the Ladder Variational Autoencoder (LVAE).

In both models the latent variables are hierarchically defined and conditioned on the preceding stochastic layer.

LVAEs improve upon DLGMs via implementation of a top-down hierarchy both in the generative and inference model.

The approximate posterior is computed via a precisionweighted update of the approximate likelihood (i.e., the inference model) and prior (i.e., the generative model).

Similarly, the PixelVAE BID16 incorporates a hierarchical latent space decomposition and uses an autoregressive decoder.

BID32 show under mild conditions that straightforward stacking of latent variables (as is done e.g. in LVAE and PixelVAE) can be ineffective, because the latent variables that are not directly conditioned on the observation variable become inactive.

Due to the nature of the sequential problem domain, our approach differs in the crucial aspects that STCNs use dynamic, i.e., conditional, priors BID7 at every level.

Moreover, the hierarchy is not only implicitly defined by the network architecture but also explicitly defined by the information content, i.e., receptive field size.

Dieng et al. FORMULA0 both theoretically and empirically show that using skip connections from the latent variable to every layer of the decoder increases mutual information between the latent and observation variables.

Similar to BID10 in STCN-dense, we introduce skip connections from all latent variables to the output.

In STCN the model is expected to encode and propagate the information through its hierarchy.

BID30 suggest using autoregressive TCN decoders to remedy the posterior collapse problem observed in language modeling with LSTM decoders BID4 .

BID29 and BID9 use TCN decoders conditioned on discrete latent variables to model audio signals.

Stochastic RNN architectures mostly vary in the way they employ the latent variable and parametrize the approximate posterior for variational inference.

BID7 and BID3 use the latent random variable to capture high-level information causing the variability observed in sequential data.

Particularly BID7 shows that using a conditional prior rather than a standard Gaussian distribution is very effective in sequence modeling.

In BID12 BID14 BID25 ), the inference model, i.e., the approximate posterior, receives both the past and future summaries of the sequence from the hidden states of forward and backward RNN cells.

The KL-divergence term in the objective enforces the model to learn predictive latent variables in order to capture the future states of the sequence.

BID21 's SWaveNet is most closely related to ours.

SWaveNet also introduces latent variables into TCNs.

However, in SWaveNet the deterministic and stochastic units are coupled which may prevent stacking of larger numbers of TCN blocks.

Since the number of stacked dilated convolutions determines the receptive field size, this directly correlates with the model capacity.

For example, the performance of SWaveNet on the IAM-OnDB dataset degrades after stacking more than 3 stochastic layers BID21 , limiting the model to a small receptive field.

In contrast, we aim to preserve the flexibility of stacking dilated convolutions in the base TCN.

In STCNs, the deterministic TCN units do not have any dependency on the stochastic variables (see FIG0 ) and the ratio of stochastic to deterministic units can be adjusted, depending on the task.

In this paper we proposed STCNs, a novel auto-regressive model, combining the computational benefits of convolutional architectures and expressiveness of hierarchical stochastic latent spaces.

We have shown the effectivness of the approach across several sequence modelling tasks and datasets.

The proposed models are trained via optimization of the ELBO objective.

Tighter lower bounds such as IWAE BID5 or FIVO (Maddison et al., 2017) may further improve modeling performance.

We leave this for future work.

The network architecture of the proposed model is illustrated in FIG6 .

We make only a small modification to the vanilla Wavenet architecture.

Instead of using skip connections from Wavenet blocks, we only use the latent sample zt in order to make a prediction of xt.

In STCN-dense configuration, zt is the concatenation of all latent variables in the hierarchy, i.e., zt = [z Output layer f (o) : For the IAM-OnDB and Deepwriting datasets we use 1D convolutions with ReLU nonlinearity.

We stack 5 of these layers with 256 filters and filter size 1.

DISPLAYFORM0 For TIMIT and Blizzard datasets Wavenet blocks in the output layer perform significantly better.

We stack 5 Wavenet blocks with dilation size 1.

For each convolution operation in the block we use 256 filters.

The filter size of the dilated convolution is set to 2.

The STCN-dense-large model is constructed by using 512 filters instead of 256.

l t : The number of Wavenet blocks is usually determined by the desired receptive field size.??? For the handwriting datasets K = 6 and L = 5.

In total we have 30 Wavenet blocks where each convolution operation has 256 filters with size 2.??? For speech datasets K = 5 and L = 5.

In total we have 25 Wavenet blocks where each convolution operation has 256 filters with size 2.

The large model configuration uses 512 filters.

p and f

q : The number of stochastic layers per task is given by L.

We used [32, 16, 8, 5 , 2] dimensional latent variables for the handwriting tasks.

It is [256, 128, 64, 32, 16] for speech datasets.

Note that the first entry of the list corresponds to z 1 .The mean and sigma parameters of the Normal distributions modeling the latent variables are calculated by the f (l) p and f (l) q networks.

We stack 2 1D convolutions with ReLU nonlinearity and filter size 1.

The number of filters are the same as the number of Wavenet block filters for the corresponding task.

Finally, we clamped the latent sigma predictions between 0.001 and 5.

In all STCN experiments we applied KL annealing.

In all tasks, the weight of the KL term is initialized with 0 and increased by 1 ?? e ???4 at every step until it reaches 1.The batch size was 20 for all datasets except for Blizzard where it was 128.We use the ADAM optimizer with its default parameters and exponentially decay the learning rate.

For the handwriting datasets the learning rate was initialized with 5 ?? e ???4 and followed a decay rate of 0.94 over 1000 decay steps.

On the speech datasets it was initialized with 1 ?? e ???3 and decayed with a rate of 0.98.

We applied early stopping by measuring the ELBO performance on the validation splits.

We implement STCN models in Tensorflow BID0 .

Our code and models achieving the SOA results are available at https://ait.ethz.ch/projects/2019/stcn/.

Here we provide the extended results table with Normal observation model entries for available models.

Table 4 : Average log-likelihood per sequence on TIMIT, Blizzard, IAM-OnDB and Deepwriting datasets. (Normal) and (GMM) stand for unimodal Gaussian or multi-modal Gaussian Mixture Model (GMM) as the observation model BID15 BID7 .

Asterisks * indicate that we used our re-implementation only for the Deepwriting dataset.

<|TLDR|>

@highlight

We combine the computational advantages of temporal convolutional architectures with the expressiveness of stochastic latent variables.