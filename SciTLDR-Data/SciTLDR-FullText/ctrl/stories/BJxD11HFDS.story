The complex world around us is inherently multimodal and sequential (continuous).

Information is scattered across different modalities and requires multiple continuous sensors to be captured.

As machine learning leaps towards better generalization to real world, multimodal sequential learning becomes a fundamental research area.

Arguably,  modeling arbitrarily distributed spatio-temporal dynamics within and across modalities is the biggest challenge in this research area.

In this paper, we present a new transformer model, called the Factorized Multimodal Transformer (FMT) for multimodal sequential learning.

FMT inherently models the intramodal and intermodal (involving two or more modalities) dynamics within its multimodal input in a factorized manner.

The proposed factorization allows for increasing the number of self-attentions to better model the multimodal phenomena at hand; without encountering difficulties during training (e.g. overfitting) even on relatively low-resource setups.

All the attention mechanisms within FMT have a full time-domain receptive field which allows them to asynchronously capture long-range multimodal dynamics.

In our experiments we focus on datasets that contain the three commonly studied modalities of language, vision and acoustic.

We perform a wide range of experiments, spanning across 3 well-studied datasets and 21 distinct labels.

FMT shows superior performance over previously proposed models, setting new state of the art in  the studied datasets.

In many naturally occurring scenarios, our perception of the world is multimodal.

For example, consider multimodal language (face-to-face communication), where modalities of language, vision and acoustic are seamlessly used together for communicative intent (Kottur et al., 2019) .

Such scenarios are widespread in everyday life, where continuous sensory perceptions form multimodal sequential data.

Each modality within multimodal data exhibits exclusive intramodal dynamics, and presents a unique source of information.

Modalities are not fully independent of each other.

Relations across two (bimodal) or more (trimodal, . . . ) of them form intermodal dynamics; often asynchronous spatio-temporal dynamics which bind modalities together .

Learning from multimodal sequential data has been an active, yet challenging research area within the field of machine learning (Baltrušaitis et al., 2018) .

Various approaches relying on graphical models or RNNs have been proposed for multimodal sequential learning.

Transformer models are a new class of neural models that rely on a carefully designed non-recurrent architecture for sequential modeling (Vaswani et al., 2017) .

Their superior performance is attributed to a self-attention mechanism, which is uniquely capable of highlighting related information across a sequence.

This self-attention is a particularly appealing mechanism for multimodal sequential learning, as it can be modified into a strong neural component for finding relations between different modalities (the cornerstone of this paper).

In practice, numerous such relations may simultaneously exist within multimodal data, which would require increasing the number of attention units (i.e. heads).

Increasing the number of attentions in an efficient and semantically meaningful way inside a transformer model, can boost the performance in modeling multimodal sequential data.

In this paper, we present a new transformer model for multimodal sequential learning, called Factorized Multimodal Transformer (FMT) .

FMT is capable of modeling asynchronous intramodal and intermodal dynamics in an efficient manner, within one single transformer network.

It does so by specifically accounting for possible sets of interactions between modalities (i.e. factorizing based on combinations) in a Factorized Multimodal Self-attention (FMS) unit.

We evaluate the performance of FMT on multimodal language: a challenging type of multimodal data which exhibits idiosyncratic and asynchronous spatio-temporal relations across language, vision and acoustic modalities.

FMT is compared to previously proposed approaches for multimodal sequential learning over multimodal sentiment analysis (CMU-MOSI) (Zadeh et al., 2016) , multimodal emotion recognition (IEMOCAP) (Busso et al., 2008) , and multimodal personality traits recognition (POM) (Park et al., 2014) .

The related works to studies in this paper fall into two main areas.

Modeling multimodal sequential data is among the core research areas within the field of machine learning.

In this area, previous work can be classified into two main categories.

The first category of models, and arguably the simplest, are models that use early or late fusion.

Early fusion uses feature concatenation of all modalities into a single modality.

Subsequently, the multimodal sequential learning task is treated as a unimodal one and tackled using unimodal sequential models such as Hidden Markov Models (HMMs) (Baum & Petrie, 1966) , Hidden Conditional Random Fields (HCRFs) , and RNNs (e.g. LSTMs Hochreiter & Schmidhuber (1997) ).

While such models are often successful for real unimodal data (i.e. not feature concatenated multimodal data), they lack the necessary components to deal with multimodal data often causes suboptimal performance (Xu et al., 2013) .

Contrary to early fusion which concatenates modalities at input level, late fusion models have relied on learning ensembles of weak classifiers from different modalities (Snoek et al., 2005; Vielzeuf et al., 2017; Nojavanasghari et al., 2016) .

Hybrid methods have also been used to combine early and late fusion together (Wu et al., 2019; Nguyen & Okatani, 2018; Lazaridou et al., 2015; Hu et al., 2017) .

The second category of models comprise of models specifically designed for multimodal data.

Multimodal variations of graphical models have been proposed, including Multi-view HCRFs where the potentials of the HCRF are changed to facilitate multiple modalities Song et al. (2012; .

Multimodal models based on LSTMs include Multi-view LSTMs Rajagopalan et al. (2016) , Memory Fusion Network (Zadeh et al., 2018a) with its recurrent and graph variants Zadeh et al., 2018c) , as well as Multi-attention Recurrent Networks (Zadeh et al., 2018b) .

Studies have also proposed generic fusion techniques that can be used in various models including Tensor Fusion and its approximate variants (Liang et al.; , as well as Compact Bilinear Pooling (Gao et al., 2015; Fukui et al., 2016; Kim et al., 2016) .

Many of these models, from both first and second categories, are used as baselines in this paper.

Transformer is a non-recurrent neural architecture designed for modeling sequential data (Vaswani et al., 2017) .

It has shown superior performance across multiple NLP tasks when compared to RNN-based or convolutional architectures (Devlin et al., 2018; Vaswani et al., 2017) .

This superior performance of Transformer model is largely credited to a self-attention; a neural component that allows for efficiently extracting both short and long-range dependencies within its input sequence space.

Transformer models have been successfully applied to various areas within machine learning including NLP and computer vision (Yang et al., 2019; Parmar et al., 2018; Alsentzer et al., 2019) .

Extending transformer to multimodal domains, specially for structured multimodal sequences is relatively understudied; with the previous works mainly focusing on using transformer models for modality alignment using cross-modal links between single transformers for each modality (Tsai et al., 2019) .

In this section, we outline the proposed Factorized Multimodal Transformer 1 (FMT).

Figure 1 shows the overall structure of the FMT model.

The input first goes through an embedding layer, followed by multiple Multimodal Transformer Layers (MTL) .

Each MTL consists of multiple Factorized Multimodal Self-attentions (FMS).

FMS explicitly accounts for intramodal and intermodal factors within its multimodal input.

S1 and S2 are two summarization networks.

They are necessary components of FMT which allow for increasing the number of attentions efficiently, without overparameterization of the FMT.

Layer (MTL) Consider a multimodal sequential dataset with constituent modalities of language, vision and acoustic.

The modalities are denoted as {L, V, A} from hereon for abbreviation.

After resampling using a reference clock, modalities can follow the same frequency .

Essentially, this resampling is often based on word timestamps (i.e. word alignment).

Subsequently, the dataset can be denoted as:

x i ∈ R Ti×dx , y i ∈ R dy are the inputs and labels.

is a triplet of language, visual and audio inputs for timestamp t in i-th datapoint.

N is the total number of samples within the dataset, and T i the total number of timestamps within i-th datapoint.

Zero paddings (on the left) can be used to unify the length of all sequences to a desired fixed length

A denotes the dimensionality of input at each timestep, which in turn is equal to the sum of dimensionality of each modality.

d y denotes the dimensionality of the associated labels of a sequence.

At the first step within the FMT model, each modality is passed to a unimodal embedding layer with the operation

Positional embeddings are also added to the input at this stage.

The output of the embeddings collectively form

We denote the dimensionality of this output as e x = e L + e V + e A .

After the initial embedding, FMT now consists of a stack of Multimodal Transformer Layers (MTL).

MTL 1) captures factorized dynamics within multimodal data in parallel, and 2) aligns the timeasynchronous information both within and across modalities.

Both of these are achieved using multiple Factorized Multimodal Self-attentions (FMS), each of which has multiple specialized selfattentions inside.

The high dimensionality of the intermediate attention outputs within MTL and FMS is controlled using two distinct summarization networks.

The continuation of this section provides detailed explanation of the inner-operations of MTL.

·,i) denote the input to the k-th MTL.

We assume a total of K MTLs in a FMT (indexed 0 . . .

K − 1), with k = 0 being the output of the embedding layer (input to k = 0 MTL).

The input of MTL, immediately goes through one/multiple 2 Factorized 1 Code: github.com/removed-for-blind-review, Public Data: https://github.com/A2Zadeh/CMUMultimodalSDK 2 Multiple FMS have the same time-domain receptive field, which is equal to the length of the input.

This is contrary to the implementations of the transformer model that split the sequence based on number of attention heads.

Output (

The grayed areas are for demonstration purposes, and not a part of the implementation.

Figure 2 .

For 3 modalities 3 , there exist 7 distinct attentions inside a single FMS unit.

Each attention has a unique receptive field with respect to modalities f ∈ F = {L,V,A,LV,LA,VA,LVA}; essentially denoting the modalities visible to the attention.

Using this factorization, FMS explicitly accounts for possible unimodal, bimodal and trimodal interactions existing within the multimodal input space.

All attentions within a FMS extend to the length of the sequence, and therefore can extract asynchronous relations within and across modalities.

For f ∈ F , each attention within a single FMS unit is controlled by the Key K f , Query Q f , and Value V f all with dimensionality R T ×T ; parameterized respectively using affine maps W K f , W Q f , and W V f .

After the attention is applied using Key, Query and Value operations (Vaswani et al., 2017) , the output of each of the attentions goes through a residual addition with its perceived input (input in the attention receptive field), followed by a normalization.

The output of the FMS contains the aligned and extracted information from the unimodal, bimodal and trimodal factors.

This output is high-dimensional; essentially R 4×T ×ex (each dimension within input of shape T ×e x is present in 4 factors).

Our goal is to reduce this high-dimensional data using a mapping from R 4×T ×ex → R T ×ex .

Without overparameterizing the FMS, in practice, we observed this mapping can be efficiently done using a simple 1D convolutional network S1 M∈{L,V,A} (·); R 4 → R. Internally, S1(·) maps its input to multiple layers of higher dimensions and subsequently to R. Using language as an example, S1 L moves across language modality dimensions e L for t = 1 . . .

T and summarizes the information across all the factors.

The output of this summarization applied on all modality dimensions and timesteps, is the output of FMS, which has the dimensionality R T ×ex .

In practice, there can be various possible unimodal, bimodal or trimodal interactions within a multimodal input.

For example, consider multiple sets of important interactions between L and V (e.g. smile + positive word, as well as eyebrows up + excited phrase), all of which need to be highlighted and extracted.

A single FMS may not be able to highlight all these interactions without diluting its intrinsic attentions.

Multiple FMS can be used inside a MTL to efficiently extract diverse multimodal interactions existing in the input data 4 .

Consider a total of U FMS units inside a MTL.

The output of each FMS goes through a feedforward network (for each timestamp t of the FMS output).

73.9/-73.4/-1.040 0.633 MARN (Zadeh et al., 2018b) 77.1/-77.0/-0.968 0.625 MFN (Zadeh et al., 2018a) 77.4/-77.3/-0.965 0.632 RMFN 78.4/-78.0/-0.922 0.681 RAVEN (Wang et al., 2018) 78.0/--/-0.915 0.691 MulT (Tsai et al., 2019) -/83.0 -/82.8 0.87 0.698 FMT (ours) 81.5/83.5 81.4/83.5 0.837 0.744 Table 1 : FMT achieves superior performance over baseline models for CMU-MOSI dataset (multimodal sentiment analysis).

We report BA (binary accuracy) and F1 (both higher is better), MAE (Mean-absolute Error, lower is better), and Corr (Pearson Correlation Coefficient, higher is better).

For BA and F1, we report two numbers: the number on the left side of "/" is calculated based on approach taken by Zadeh et al. (2018b) , and the right side is by Tsai et al. (2019) .

The output of this feedfoward network is residually added with its input, and subsequently normalized.

The feedforward network is the same across all U FMS units and timestamps t. Subsequently, the dimensionality of the output of the normalizations collectively is R U ×T ×ex .

Similar to operations performed by S1, a secondary summarization network S2 M∈{L,V,A} (·); R U → R can be used here.

S2 is also a 1D convolutional network that moves across modality dimensions and different timesteps to map R U ×T ×ex to R T ×ex .

The output of the secondary summarization network is the final output of MTL, and denoted asx

be the output of last MTL in the stack.

For supervision, we feed this input one timestamp at a time as input to a Gated Recurrent Unit (GRU) (Cho et al., 2014) .

The prediction is conditioned on output at timestamp t = T of the GRU, using an affine map to d y .

In this section, we discuss the experimental methodology including tasks, datasets, computational descriptors, and comparison baselines.

The following inherently multimodal tasks (and accompanied datasets) are studied in this paper.

All the tasks are related to multimodal language: a complex and idiosyncratic sequential multimodal signal, where semantics are arbitrarily scattered across modalities (Holler & Levinson, 2019) .

The first benchmark in our experiments is multimodal sentiment analysis, where the goal is to identify a speaker's sentiment based on the speaker's display of verbal and nonverbal behaviors.

We use the well-studied CMU-MOSI (CMU Multimodal Opinion Sentiment Intensity) dataset for this purpose (Zadeh et al., 2016) .

There are a total of 2199 data points (opinion utterances) within CMU-MOSI dataset.

The dataset has real-valued sentiment intensity annotations in the range [−3, +3] .

It is considered a challenging dataset due to speaker diversity (1 video per distinct speaker), topic variations and low-resource setup.

The second benchmark in our experiments is multimodal emotion recognition, where the goal is to identify a speaker's emotions based on the speaker's verbal and nonverbal behaviors.

We use the well-studied IEMOCAP dataset (Busso et al., 2008) .

IEMOCAP consists of 151 sessions of recorded dialogues, of which there are 2 speaker's per session for a total of 302 videos across the dataset.

We perform experiments for discrete emotions (Ekman, 1992) of Happy, Sad, Angry and Neutral (no emotions) -similar to previous works (Tsai et al., 2019; Wang et al., 2018 24.1 31.0 31.5 34.5 24.6 25.6 27.6 29.1 MARN (Zadeh et al., 2018b) 29.1 33.0 --31.5 ---MFN (Zadeh et al., 2018a) 34.5 35.5 37.4 41.9 34.5 36.9 36.0 37.9 RMFN 37.4 38.4 37.4 -37.4 38.9 38.9 -MulT (Tsai et al., 2019) 34.5 34.5 36.5 38.9 37.4 36.9 37.

30.5 38.9 35.5 37.4 33.0 42.4 27.6 33.0 MARN (Zadeh et al., 2018b) 36.9 -52.2 --47.3 31.0 44.8 MFN (Zadeh et al., 2018a) 38.

Table 3 : FMT achieves superior performance over baseline models in POM dataset (multimodal personality traits recognition).

For label abbreviations please refer to Section 4.3.

MA(5,7) denotes multi-class accuracy for (5,7)-class personality labels (higher is better).

The third benchmark in our experiments is speaker trait recognition based on communicative behavior of a speaker.

It is a particularly difficult task, with 16 different speaker traits in total.

We study the POM dataset which contains 1,000 movie review videos (Park et al., 2014) .

Each video is annotated for various personality and speaker traits, specifically: Confident (Con), Passionate (Pas), Voice Pleasant (Voi), Dominant (Dom), Credible (Cre), Vivid (Viv), Expertise (Exp), Entertaining (Ent), Reserved (Res), Trusting (Tru), Relaxed (Rel), Outgoing (Out), Thorough (Tho), Nervous (Ner), Persuasive (Per) and Humorous (Hum).

The short form of these speaker traits is indicated inside the parentheses and used for the rest of this paper.

The following computational descriptors are used by FMT and baselines (all the baselines use the same descriptors in their original respective papers).

Language: P2FA forced alignment model (Yuan & Liberman, 2008 ) is used to align the text and audio at word level.

From the forced alignment, the timing of words and sentences are extracted.

Word-level alignment is used to unify the modality frequencies .

GloVe embeddings (Pennington et al., 2014) are subsequently used for word representation.

Visual: For the visual modality, the Emotient FACET (iMotions, 2017) is used to extract a set of visual features including Facial Action Units (Ekman et al., 1980) , visual indicators of emotions, and sparse facial landmarks.

Acoustic: COVAREP (Degottex et al., 2014 ) is used to extract the following features: fundamental frequency, quasi open quotient (Kane & Gobl, 2013) , normalized amplitude quotient, glottal source parameters (H1H2, Rd, Rd conf) (Drugman et al., 2012) , Voiced/Unvoiced segmenting features (VUV) (Drugman & Alwan, 2011) , maxima dispersion quotient (MDQ), the first 3 formants, parabolic spectral parameter (PSP), harmonic model and phase distortion mean (HMPDM 0-24) and deviations (HMPDD 0-12), spectral tilt/slope of wavelet responses (peak/slope), Mel Cepstral Coefficients (MCEP 0-24).

The following strong baselines are compared to FMT: MV-LSTM (Multi-view LSTM, Rajagopalan et al. (2016) (2019)).

There are fundamental distinctions between FMT and MulT, chief among them: 1) MulT consists of 6 transformers, 3 cross-modal transformers and 3 unimodal.

Naturally this increases the overall model size substantially.

FMT consists of only one transformer, with components to avoid overparameterization.

2) FMT sees interactions as undirected (unlike MulT which has L → V and V → L), and therefore semantically combines two attentions in one.

3) MulT has no trimodal factors (which are important according to Section 5).

4) MulT has no direct unimodal path (e.g. only L), as input to unimodal transformers are outputs of cross-modal transformers.

5) All FMT attentions have full time-domain receptive field, while MulT splits the input based on the heads.

In their original publication, all the models report 6 the performance over the datasets in Section 4.1, using the same descriptors discussed in Section 4.2.

The models in this paper are compared using the following performance measures (depending on the dataset): (BA) denotes binary accuracyhigher is better, (MA5,MA7) are 5 and 7 multiclass accuracy -higher is better, (F1) denotes F1 score -higher is better, (MAE) denotes the Mean-Absolute Error -lower is better, (Corr) is Pearson Correlation Coefficient -higher is better.

The hyperparameter space search for FMT (and baselines if retrained) is discussed in Appendix A.1.

The results of sentiment analysis experiments on CMU-MOSI dataset are presented in Table 1 .

FMT achieves superior performance than the previously proposed models for multimodal sentiment analysis.

We use two approaches for calculating BA and F1 based on negative vs. non-negative sentiment (Zadeh et al., 2018b) on the left side of /, and negative vs. positive (Tsai et al., 2019) on the right side.

MAE and Corr are also reported.

For multimodal emotion recognition, experiments on IEMOCAP are reported in Table 2 .

The performance of FMT is superior than other baselines for multimodal emotion recognition (with the exception of Happy emotion).

The results of experiments for personality traits recognition on POM dataset are reported in Table 3 .

We report MA5 and MA7, depending on the label.

FMT outperforms baselines across all personality traits.

We study the importance of the factorization in FMT.

We first remove the unimodal, bimodal and trimodal attentions from the FMT model, resulting in 3 alternative implementations of FMT.

demonstrates the results of this ablation experiment over CMU-MOSI dataset.

Furthermore, we use only one modality as input for FMT, to understand the importance of each modality (all other factors removed).

We also replace the summarization networks with simple vector addition operation.

All factors, modalities, and summarization components are needed for achieving best performance.

We also perform experiments to understand the effect of number of FMT units within each MTL.

Table 5 shows the performance trend for different number of FMT units.

The model with 6 number of FMS (42 attentions in total) achieves the highest performance (6 is also the highest number we experimented with).

Tsai et al. (2019) reports the best performance for CMU-MOSI dataset is achieved when using 40 attentions per cross-modal transformer (3 of each, therefore 120 attention, without counting the subsequent unimodal transformers).

FMT uses fewer number of attentions than MulT, yet achieves better performance.

We also experiment with number of heads for original transformer model (Vaswani et al., 2017) and compare to FMT (Appendix A.3).

In this paper, we presented the Factorized Multimodal Transformer (FMT) model for multimodal sequential learning.

Using a Factorized Multimodal Self-attention (FMS) within each Multimodal Transformer Layer (MTL), FMT is able to model the intra-model and inter-modal dynamics within asynchronous multimodal sequences.

We compared the performance of FMT to baselines approaches over 3 publicly available datasets for multimodal sentiment analysis (CMU-MOSI, 1 label), emotion recognition (IEMOCAP, 4 labels) and personality traits recognition (POM, 16 labels).

Overall, FMT achieved superior performance than previously proposed models across the studied datasets.

A APPENDIX

The hyperparameters of FMT include the Adam (Kingma & Ba, 2014) learning rate ({0.001, 0.0001}), structure of summarization network (randomly picked 5 architectures from {1, 2, 3} layers of conv, with kernel shapes of {2, 5, 10, 15, 20}), number of MTL layers ({4, 6, 8} except for ablation experiments which was 2 . . .

8), number of FMT units ({4, 6}, except for ablation experiment which was 1 . . .

6), e M ∈{L,V,A} ({20, 40}), dropout (0, 0.1).

The same parameters (when applicable) are used for training MulT for POM dataset (e.g. num encoder layers same as number of MTL).

Furthermore, for MulT specific hyperparameters, we use similar values as Table 5 in the original paper.

All models are trained for a maximum of 200 epochs.

The hyperparameter validation is similar to Zadeh et al. (2018b) .

We study the effect of number of MTL on FMT performance.

Table 6 shows the results of this experiment.

The best performance is achieved using 8 MTL layers (which was also the maximum layers we tried in our hyperparameter search).

In this section, we discuss the effect of increasing the number of heads on the original transformer model (OTF, Vaswani et al. (2017) ).

Please note that we implement the OTF to allow for all attention heads to have full input receptive field (from 1 . . .

T ), similar to FMT.

We increase the attention heads from 1 to 35 (after 35 does not fit on a Tesla-V100 GPU with batchsize of 20).

Table 7 shows the results of increasing number of attention heads for both models.

We observe that achieving superior performance is not a matter of increasing the attention heads.

Even using 1 FMS unit, which leads to 7 total attention, FMT achieves higher performance than counterpart OTF.

In many scenarios in nature, as well as what is currently pursued in machine learning, the number of modalities goes as high as 3 (mostly language, vision and acoustic, as studied in this paper).

This leads to 7 attentions within each FMS, well manageable for successful training of FMT as demonstrated in this paper.

However, as the number of modalities increases, the underlying multimodal phenomena becomes more challenging to model.

This causes complexities for any competitive multimodal model, regardless of their internal design.

While studying these cases are beyond the scope of this paper, due to rare nature of having more than 3 main modalities modalities, for FMT, the complexity can be managed due to the factorization in FMS.

We propose two approaches: 1) for high number of modalities, the involved factors can be reduced based on domain knowledge, the nature of the problem, and the assumed dependencies between modalities (e.g. removing factors between modalities that are deemed weakly related).

Alternatively, without making assumptions about inter-modality dependencies, a greedy approach may be taken for adding factors; an approach similar to stepwise regression (Kleinbaum et al., 1988) , iteratively adding the next most important factor.

Using these two methods, the model can cope with higher number of modalities with a controllable compromise between performance and overparameterization.

<|TLDR|>

@highlight

A multimodal transformer for multimodal sequential learning, with strong empirical results on multimodal language metrics such as multimodal sentiment analysis, emotion recognition and personality traits recognition. 