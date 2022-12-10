Given the fast development of analysis techniques for NLP and speech processing systems, few systematic studies have been conducted to compare the strengths and weaknesses of each method.

As a step in this direction we study the case of representations of phonology in neural network models of spoken language.

We use two commonly applied analytical techniques, diagnostic classifiers and representational similarity analysis, to quantify to what extent neural activation patterns encode phonemes and phoneme sequences.

We manipulate two factors that can affect the outcome of analysis.

First, we investigate the role of learning by comparing neural activations extracted from trained versus randomly-initialized models.

Second, we examine the temporal scope of the activations by probing both local activations corresponding to a few milliseconds of the speech signal, and global activations pooled over the whole utterance.

We conclude that reporting analysis results with randomly initialized models is crucial, and that global-scope methods tend to yield more consistent and interpretable results and we recommend their use as a complement to local-scope diagnostic methods.

As end-to-end architectures based on neural networks became the tool of choice for processing speech and language, there has been increased interest in techniques for analyzing and interpreting the representations emerging in these models.

A large array of analytical techniques have been proposed and applied to diverse tasks and architectures .

Given the fast development of analysis techniques for NLP and speech processing systems, relatively few systematic studies have been conducted to compare the strengths and weaknesses of each methodology and to assess the reliability and explanatory power of their outcomes in controlled settings.

This paper reports a step in this direction: as a case study, we examine the representation of phonology in neural network models of spoken language.

We choose three different models that process speech signal as input, and analyze their learned neural representations.

We use two commonly applied analytical techniques: (i) diagnostic models and (ii) representational similarity analysis to quantify to what extent neural activation patterns encode phonemes and phoneme sequences.

In our experiments, we manipulate two important factors that can affect the outcome of analysis.

One pitfall not always successfully avoided in work on neural representation analysis is the role of learning.

Previous work has shown that sometimes non-trivial representations can be found in the activation patterns of randomly initialized, untrained neural networks (Zhang and Bowman, 2018; ).

Here we investigate the representations of phonology in neural models of spoken language in light of this fact, as extant studies have not properly controlled for role of learning in these representations.

The second manipulated factor in our experiments is the scope of the extracted neural activations.

We control for the temporal scope, probing both local activations corresponding to a few milliseconds of the speech signal, as well as global activations pooled over the whole utterance.

When applied to global-scope representations, both the methods detect a robust difference between the trained and randomly initialized target models.

However we find that in our setting, RSA applied to local representations shows low correlations between phonemes and neural activation patterns for both trained and randomly initialized target models, and for one of the target models the local diagnostic classifier only shows a minor difference in the decodability of phonemes from randomly initialized versus trained network.

This highlights the importance of reporting analy-sis results with randomly initialized models as a baseline.

Many current neural models of language learn representations that capture useful information about the form and meaning of the linguistic input.

Such neural representations are typically extracted from activations of various layers of a deep neural architecture trained for a target task such as automatic speech recognition or language modeling.

A variety of analysis techniques have been proposed in the academic literature to analyze and interpret representations learned by deep learning models of language as well as explain their decisions; see and for a review.

Some of the proposed techniques aim to explain the behavior of a network by tracking the response of individual or groups of neurons to an incoming trigger (e.g., Nagamine et al., 2015; Krug et al., 2018) .

In contrast, a larger body of work is dedicated to determining what type of linguistic information is encoded in the learned representations.

This type of analysis is the focus of our paper.

Two commonly used approaches to analyzing representations are Probing techniques, or diagnostic classifiers, i.e. methods which use the activations from different layers of a deep learning architecture as input to a prediction model (e.g., Adi et al., 2017; Hupkes et al., 2018; Conneau et al., 2018) ;

Representational Similarity Analysis (RSA) borrowed from neuroscience (Kriegeskorte et al., 2008) and used to correlate similarity structures of two different representation spaces (Bouchacourt and Baroni, 2018; Abnar et al., 2019; Abdou et al., 2019) We use both techniques in our experiments to systematically compare their output.

Research on the analysis of neural encodings of language has shown that in some cases, substantial information can be decoded from activation patterns of randomly initialized, untrained recurrent networks.

It has been suggested that the dynamics of the network together with the characteristics of the input signal can result in non-random activation patterns (Zhang and Bowman, 2018) .

Using activations generated by randomly initialized recurrent networks has a history in speech recognition and computer vision.

Two betterknown families of such techniques are called Echo State Networks (ESN) (Jaeger, 2001) and Liquid State Machines (LSM) (Maass et al., 2002) .

The general approach (also known as reservoir computing) is as follows: the input signal is passed through a randomly initialized network to generate a nonlinear response signal.

This signal is then used as input to train a model to generate the desired output at a reduced cost.

We also focus on representations from randomly initialized neural models but do so in order to show how training a model changes the information encoded in the representations according to our chosen analysis methods.

Since the majority of neural models of language work with text rather than speech, the bulk of work on representation analysis has been focused on (written) word and sentence representations.

However, a number of studies analyze neural representations of phonology learned by models that receive a speech signal as their input.

As examples of studies that track responses of neurons to controled input, Nagamine et al. (2015) analyze local representations acquired from a deep model of phoneme recognition and show that both individual and groups of nodes in the trained network are selective to various phonetic features, including manner of articulation, place of articulation, and voicing.

Krug et al. (2018) use a similar approach and suggest that phonemes are learned as an intermediate representation for predicting graphemes, especially in very deep layers.

Others predominantly use diagnostic classifiers for phoneme and grapheme classification from neural representations of speech.

In one of the their experiments use a linear classifier to predict phonemes from local activation patterns of a grounded language learning model, where images and their spoken descriptions are processed and mapped into a shared semantic space.

Their results show that the network encodes substantial knowledge of phonology on all its layers, but most strongly on the lower recurrent layers.

Similarly, Belinkov and Glass (2017) use diagnostic classifiers to study the encoding of phonemes in an end-to-end ASR system with convolutional and recurrent layers, by feeding local (frame-based) representations to an MLP to predict a phoneme label.

They show that phonological information is best represented in lowest input and convolutional layers and to some extent in low-to-middle recurrent layers.

extend their previous work to multiple languages (Arabic and English) and different datasets, and show a consistent pattern across languages and datasets where both phonemes and graphemes seem to be encoded best in the middle recurrent layers.

None of these studies report on phoneme classification from randomly initialized versions of their target models, and none use global (i.e., utterancelevel) representations in their analyses.

In this section we first describe the speech models which are the targets of our analyses, followed by a discussion of the methods used here to carry out these analyses.

We will release source code to run all our analyses on the publication of this paper.

We tested the analysis methods on three target models trained on speech data.

The first model is a transformer model (Vaswani et al., 2017) trained on the automatic speech recognition (ASR) task.

More precisely, we used a pretrained joint CTC-Attention transformer model from the ESPNet toolkit (Watanabe et al., 2018) , trained on the Librispeech dataset (Panayotov et al., 2015) .

1 The architecture is based on the hybrid CTC-Attention decoding scheme presented by Watanabe et al. (2017) but adapted to the transformer model.

The encoder is composed of two 2D convolutional layers (with stride 2 in both time and frequency) and a linear layer, followed by 12 transformer layers, while the decoder has 6 such layers.

The convolutional layers use 512 channels, which is also the output dimension of the 1 We used ESPnet code from commit 8fdd8e9 with the pretrained model available from https://drive.google.

com/open?id=1BtQvAnsFvVi-dp_qsaFP7n4A_ 5cwnlR6 linear and transformer layers.

The dimension of the flattened output of the two convolutional layers (along frequencies and channel) is then 20922 and 10240 respectively: we omit these two layers in our analyses due to their excessive size.

The input to the model is made of a spectrogram with 80 coefficients and 3 pitch features, augmented with the SpecAugment method (Park et al., 2019) .

The output is composed of 5000 SentencePiece subword tokens (Kudo and Richardson, 2018) .

The model is trained for 120 epochs using the optimization strategy from Vaswani et al. (2017) , also known as Noam optimization.

Decoding is performed with a beam of size 60 for reported word error rates of 2.6% and 5.7% on the test set (for the clean and other subsets respectively).

The Visually Grounded Speech (VGS) model is trained on the task of matching images with their corresponding spoken captions, first introduced by Harwath and Glass (2015) and Harwath et al. (2016) .

We use the architecture of Merkx et al. (2019) which implemented several improvements over the RNN model of , and train it on the Flickr8K Audio Caption Corpus (Harwath and Glass, 2015) .

The speech encoder consists of one 1D convolutional layer (with 64 output channels) which subsamples the input by a factor of two, and four bidirectional GRU layers (each of size 2048) followed by a self-attention-based pooling layer.

The image encoder uses features from a pre-trained ResNet-152 model (He et al., 2016) followed by a linear projection.

The loss function is a margin-based ranking objective.

Following Merkx et al. (2019) we trained the model using the Adam optimizer (Kingma and Ba, 2015) with a cyclical learning rate schedule (Smith, 2017) .

The input are MFCC features with total energy and delta and double-delta coefficients with combined size 39.

This model is a middle ground between the two previous ones.

It is trained as a speech recognizer similarly to the transformer model but the architecture of the encoder follows the RNN-VGS model (except that the recurrent layers are one-directional in order to fit the model in GPU memory).

The last GRU layer of the encoder is fed to the attention-based decoder from Bahdanau et al. (2015) , here composed of a single layer of 1024 GRU units.

The model is trained with the Adadelta optimizer (Zeiler, 2012) .

The input features are identical to the ones used for the VGS model; it is also trained on the Flickr8k dataset spoken caption data, using the original written captions as transcriptions.

The architecture of this model is not optimized for the speech recognition task: rather it is designed to be as similar as possible to the RNN-VGS model while still performing reasonably on speech recognition.

We consider two analytical approaches:

• Diagnostic model is a simple, often linear, classifier or regressor trained to predict some information of interest given neural activation patterns.

To the extent that the model successfuly decodes the information, we conclude that this information is present in the neural representations.

• Representational similarity analysis (RSA) is a second-order approach where similarities between pairs of some stimuli are measured in two representation spaces: e.g. neural activation pattern space and a space of symbolic linguistic representations such as sequences of phonemes or syntax trees (see .

Then the correlation between these pairwise similarity measurements quantifies how much the two representations are aligned.

The diagnostic models have trainable parameters while the RSA-based models do not, except when using a trainable pooling operation.

We also consider two ways of viewing activation patterns in hidden layers as representations:

• Local representations at the level of a single frame or time-step;

• Global representations at the level of the whole utterance.

Combinations of these two facets give rise to the following concrete analysis models.

Local diagnostic classifier.

We use single frames of input (MFCC or spectrogram) features, or activations at a single timestep as input to a logistic diagnostic classifier which is trained to predict the phoneme aligned to this frame or timestep.

Local RSA.

We compute two sets of similarity scores.

For neural representations, these are cosine similarities between neural activations from pairs of frames.

For phonemic representations our similarities are binary, indicating whether a pair of frames are labeled with the same phoneme.

Pearson's r coefficient computed against a binary variable, as in our setting, is also known as point biserial correlation.

Global diagnostic classifier.

We train a linear diagnostic classifier to predict the presence of phonemes in an utterence based on global (pooled) neural activations.

For each phoneme j the predicted probability that it is present in the utterance with representation h is denoted as P(j|h) and computed as:

where Pool is one of the pooling function in Section 3.2.1.

Global RSA.

We compute pairwise similarity scores between global (pooled; see Section 3.2.1) representations and measure Pearson's r with the pairwise string similarities between phonemic transcriptions of utterances.

We define string similarity as:

where | · | denotes string length and Levenshtein is the string edit distance.

The representations we evaluate are sequential: sequences of input frames, or of neural activation states.

In order to pool them into a single global representation of the whole utterance we test two approaches.

Mean pooling.

We simply take the mean for each feature along the time dimension.

Attention-based pooling.

Here we use a simple self-attention operation with parameters trained to optimize the score of interest, i.e. the RSA score or the error of the diagnostic classifier.

The attentionbased pooling operator performs a weighted average over the positions in the sequence, using scalar weights.

The pooled utterance representation Pool(h) is defined as:

with the weights α computed as:

where w are learnable parameters, and h t is an input or activation vector at position t. 2

For RSA we use Pearson's r to measure how closely the activation similarity space corresponds to the phoneme or phoneme string similarity space.

For the diagnostic classifiers we use the relative error reduction (RER) over the majority class baseline to measure how well phoneme information can be decoded from the activations.

Effect of learning In order to be able to assess and compare how sensitive the different methods are to the effect of learning on the activation patterns, it is important to compare the score on the trained model to that on the randomly initialized model; we thus always display the two jointly.

We posit that a desirable property of an analytical method is that it is sensitive to the learning effect, and that the scores on trained versus randomly initialized models are clearly separated.

Coefficient of partial determination Correlation between similarity structures of two representational spaces can, in principle, be partly due to the fact that both these spaces are correlated to a third space.

For example, were we to get a high value for global RSA for one of the top layers of the RNN-VGS model, we might suspect that this is due to the fact that string similarities between phonemic transcriptions of captions are correlated to visual similarities between their corresponding images, rather than due to the layer encoding phoneme strings.

In order to control for this issue, we can carry out RSA between two spaces while controling for the third, confounding, similarity space.

We do this by computing the coefficient of partial determination defined as the relative reduction in error caused by including variable X in a linear regression model 2 Note that the visually grounded speech models of ; Chrupała (2019); Merkx et al. (2019) use similar mechanisms to aggregate the activations of the final RNN layer; here we use it as part of the analytical method to pool any sequential representation of interest.

A further point worth noting is that we use scalar weights αt and apply a linear model for learning them in order to keep the analytic model simple and easy to train consistently.

for Y :

where e Y ∼X+Z is the sum squared error of the model with all variables, and e Y ∼Z is the sum squared error of the model with X removed.

Given the scenario above with the confounding space being visual similarity, we identify Y as the pairwise similarities in phoneme string space, X as the similarities in neural activation space, and Z as similarities in the visual space.

The visual similarities are computed via cosine similarity on the image feature vectors corresponding to the stimulus utterances.

All analytical methods are implemented in Pytorch (Paszke et al., 2019) .

The diagnostic classifiers are trained using Adam with learning rate schedule which is scaled by 0.1 after 10 epochs with no improvement in accuracy.

We terminate training after 50 epochs with no improvement.

Global RSA with attention-based pooling is trained using Adam for 60 epochs with a fixed learning rate (0.001).

For all trainable models we snapshot model parameters after every epoch and report the results for the epoch with best validation score.

In all cases we sample half of the available data for training (if applicable), holding out the other half for validation.

Sampling data for local RSA.

When computing RSA scores it is common practice in neuroscience research to use the whole upper triangular part of the matrices containing pairwise similarity scores between stimuli, presumably because the number of stimuli is typically small in that setting.

In our case the number of stimuli is very large, which makes using all the pairwise similarities computationally taxing.

More importantly, when each stimulus is used for computing multiple similarity scores, these scores are not independent, and score distribution changes with the number of stimuli.

We therefore use an alternative procedure where each stimulus is sampled without replacement and used only in a single similarity calculation.

Figures 1-3 display the outcome of analyzing our target models.

All three figures are organized in a 2 × 3 matrix of panels, with the top row showing the diagnostic methods and the bottom row the RSA methods; the first column corresponds to local scope; column two and three show global scope with mean and attention pooling respectively.

The data points are displayed in the order of the hierarchy of layers for each architecture, starting with the input (layer id = 0).

In all the reported experiments, the score of the diagnostic classifiers corresponds to relative error reduction (RER), whereas for RSA we show Pearson's correlation coefficient.

Figure 4 shows the results of global RSA with mean pooling on the RNN-VGS target model, while controling for visual similarity as a confound.

We will discuss the patterns of results observed for each model separately in the following sections.

As can be seen in Figure 1 , most reported experiments (with the exception of the local RSA) suggest that phonemes are best encoded in pre-final layers of the deep network.

The results also show a strong impact of learning on the predictions of the analytical methods, as is evident by the difference between the performance using representations of the trained versus randomly initialized models.

Local RSA shows low correlation values overall, and does not separate the trained versus random conditions well.

Most experimental findings displayed in Figure 2 suggest that phonemes are best encoded in RNN layers 3 and 4 of the VGS model.

They also show that the representations extracted from the trained model encode phonemes more strongly than the ones from the random version of the model.

However, the impact of learning is more salient with global than local scope: the scores of both local classifier and local RSA on random vs. trained representations are close to each other for all layers.

For the global representations the performance on trained representations quickly diverges from the random representations from the first RNN layer onward.

Furthermore, as demonstrated in Figure 4 , for top RNN layers of this architecture, the correlation between similarities in the neural activation space and the similarities in the phoneme string space is not solely due to both being correlated to visual similarities: indeed similarities in activation space contribute substantially to predicting string similarities, over and above the visual similarities.

The overall qualitative patterns for this target model are the same as for RNN-VGS.

The absolute scores for the global diagnostic variants are higher, and the curves steeper, which may reflect that the objective for this target model is more closely aligned with encoding phonemes than in the case of RNN-VGS.

Here we discuss the impact of each factor in the outcome of our analyses.

Choice of method.

The choice of RSA versus diagnostic classifier interacts with scope, and thus these are better considered as a combination.

Specifically, local RSA as implemented in this study shows only weak correlations between neural activations and phoneme labels.

It is possibly related to the range restriction of point biserial correlation with unbalanced binary variables.

Impact of learning.

Applied to the global representations, both analytical methods are equally sensitive to learning.

The results on random vs. trained representations for both methods start to diverge noticeably from early recurrent layers.

The separation for the local diagnostic classifiers is weaker for the RNN models.

Representation scope.

Although the temporal scale of the extracted representations (especially those of spoken language) has not received much attention and scrutiny, our experimental findings suggest that this is an important choice.

Specifically, global representations seem to be more sensitive to learning, and more consistent across different analysis methods.

Results with attention-based learned pooling are in general more erratic than with mean pooling.

This reflects the fact that analytical models which incorporate learned pooling are more difficult to optimize and require more careful tuning compared to mean pooling.

Given the above findings, we now give tentative recommendations regarding how to carry out representational analyses of neural models.

• Analyses should be run on randomly initialized target models.

We saw that in most cases scores on such models are substantially above zero, and in some cases relatively close to scores on trained models.

• It is unwise to rely on a single analytical approach, even a widely used one such as the local diagnostic classifier.

With solely this method we would have concluded that, in RNN models, learning has only a weak effect on the encoding of phonology.

• Global methods applied to pooled representations should be considered as a complement to standard local diagnostic methods.

In our experiments they show more consistent results.

We carried out a systematic study of analysis methods for neural models of spoken language and offered some suggestions on best practices in this endeavor.

Nevertheless our work is only a first step, and several limitations remain.

The main challenge is that it is often difficult to completely control for the many factors of variation present in the target models, due to the fact that a particular objective function, or even a dataset, may require relatively important architectural modifications.

In future we plan to sample target models with a larger number of plausible combinations of factors.

Likewise, a choice of an analytical method may often entail changes in other aspects of the analysis: for example unlike a global diagnostic classifier, global RSA captures the sequential order of phonemes.

In future we hope to further disentangle these differences.

<|TLDR|>

@highlight

We study representations of phonology in neural network models of spoken language with several variants of analytical techniques.