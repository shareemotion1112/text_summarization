We propose a simple technique for encouraging generative RNNs to plan ahead.

We train a ``backward'' recurrent network to generate a given sequence in reverse order, and we encourage states of the forward model to predict cotemporal states of the backward model.

The backward network is used only during training, and plays no role during sampling or inference.

We hypothesize that our approach eases modeling of long-term dependencies by implicitly forcing the forward states to hold information about the longer-term future (as contained in the backward states).

We show empirically that our approach achieves 9% relative improvement for a speech recognition task, and achieves significant improvement on a COCO caption generation task.

Recurrent Neural Networks (RNNs) are the basis of state-of-art models for generating sequential data such as text and speech.

RNNs are trained to generate sequences by predicting one output at a time given all previous ones, and excel at the task through their capacity to remember past information well beyond classical n-gram models BID6 BID27 .

More recently, RNNs have also found success when applied to conditional generation tasks such as speech-to-text BID9 , image captioning BID61 and machine translation .RNNs are usually trained by teacher forcing: at each point in a given sequence, the RNN is optimized to predict the next token given all preceding tokens.

This corresponds to optimizing one-stepahead prediction.

As there is no explicit bias toward planning in the training objective, the model may prefer to focus on the most recent tokens instead of capturing subtle long-term dependencies that could contribute to global coherence.

Local correlations are usually stronger than long-term dependencies and thus end up dominating the learning signal.

The consequence is that samples from RNNs tend to exhibit local coherence but lack meaningful global structure.

This difficulty in capturing long-term dependencies has been noted and discussed in several seminal works (Hochreiter, 1991; BID6 BID27 BID45 .Recent efforts to address this problem have involved augmenting RNNs with external memory BID14 BID18 BID22 , with unitary or hierarchical architectures BID0 BID51 , or with explicit planning mechanisms BID23 .

Parallel efforts aim to prevent overfitting on strong local correlations by regularizing the states of the network, by applying dropout or penalizing various statistics BID41 BID64 BID15 BID32 BID39 .

Figure 1: The forward and the backward networks predict the sequence s = {x 1 , ..., x 4 } independently.

The penalty matches the forward (or a parametric function of the forward) and the backward hidden states.

The forward network receives the gradient signal from the log-likelihood objective as well as L t between states that predict the same token.

The backward network is trained only by maximizing the data log-likelihood.

During the evaluation part of the network colored with orange is discarded.

The cost L t is either a Euclidean distance or a learned metric ||g(h DISPLAYFORM0 with an affine transformation g. Best viewed in color.

In this paper, we propose TwinNet, 1 a simple method for regularizing a recurrent neural network that encourages modeling those aspects of the past that are predictive of the long-term future.

Succinctly, this is achieved as follows: in parallel to the standard forward RNN, we run a "twin" backward RNN (with no parameter sharing) that predicts the sequence in reverse, and we encourage the hidden state of the forward network to be close to that of the backward network used to predict the same token.

Intuitively, this forces the forward network to focus on the past information that is useful to predicting a specific token and that is also present in and useful to the backward network, coming from the future (Fig. 1) .In practice, our model introduces a regularization term to the training loss.

This is distinct from other regularization methods that act on the hidden states either by injecting noise BID32 or by penalizing their norm BID31 BID39 , because we formulate explicit auxiliary targets for the forward hidden states: namely, the backward hidden states.

The activation regularizer (AR) proposed by BID39 , which penalizes the norm of the hidden states, is equivalent to the TwinNet approach with the backward states set to zero.

Overall, our model is driven by the intuition (a) that the backward hidden states contain a summary of the future of the sequence, and (b) that in order to predict the future more accurately, the model will have to form a better representation of the past.

We demonstrate the effectiveness of the TwinNet approach experimentally, through several conditional and unconditional generation tasks that include speech recognition, image captioning, language modelling, and sequential image generation.

To summarize, the contributions of this work are as follows:• We introduce a simple method for training generative recurrent networks that regularizes the hidden states of the network to anticipate future states (see Section 2);• The paper provides extensive evaluation of the proposed model on multiple tasks and concludes that it helps training and regularization for conditioned generation (speech recognition, image captioning) and for the unconditioned case (sequential MNIST, language modelling, see Section 4);• For deeper analysis we visualize the introduced cost and observe that it negatively correlates with the word frequency (more surprising words have higher cost).

Given a dataset of sequences S = {s 1 , . . .

, s n }, where each s k = {x 1 , . . .

, x T k } is an observed sequence of inputs x i ∈ X , we wish to estimate a density p(s) by maximizing the log-likelihood of the observed data L = n i=1 log p(s i ).

Using the chain rule, the joint probability over a sequence x 1 , . . . , x T decomposes as: DISPLAYFORM0 This particular decomposition of the joint probability has been widely used in language modeling BID7 BID40 and speech recognition BID5 .

A recurrent neural network is a powerful architecture for approximating this conditional probability.

At each step, the RNN updates a hidden state h f t , which iteratively summarizes the inputs seen up to time t: h DISPLAYFORM1 where f symbolizes that the network reads the sequence in the forward direction, and Φ f is typically a non-linear function, such as a LSTM cell BID27 or a GRU .

Thus, h f t forms a representation summarizing information about the sequence's past.

The prediction of the next symbol x t is performed using another non-linear transformation on top of h DISPLAYFORM2 , which is typically a linear or affine transformation (followed by a softmax when x t is a symbol).

The basic idea of our approach is to encourage h f t to contain information that is useful to predict x t and which is also compatible with the upcoming (future) inputs in the sequence.

To achieve this, we run a twin recurrent network that predicts the sequence in reverse and further require the hidden states of the forward and the backward networks to be close.

The backward network updates its hidden state according to: DISPLAYFORM3 and predicts DISPLAYFORM4 using information only about the future of the sequence.

Thus, h f t and h b t both contain useful information for predicting x t , coming respectively from the past and future.

Our idea consists in penalizing the distance between forward and backward hidden states leading to the same prediction.

For this we use the Euclidean distance (see Fig. 1 ): DISPLAYFORM5 where the dependence on x is implicit in the definition of h f t and h b t .

The function g adds further capacity to the model and comes from the class of parameterized affine transformations.

Note that this class includes the identity tranformation.

As we will show experimentally in Section 4, a learned affine transformation gives more flexibility to the model and leads to better results.

This relaxes the strict match between forward and backward states, requiring just that the forward hidden states are predictive of the backward hidden states.

The total objective maximized by our model for a sequence s is a weighted sum of the forward and backward log-likelihoods minus the penalty term, computed at each time-step: DISPLAYFORM0 where α is an hyper-parameter controlling the importance of the penalty term.

In order to provide a more stable learning signal to the forward network, we only propagate the gradient of the penalty term through the forward network.

That is, we avoid co-adaptation of the backward and forward networks.

During sampling and evaluation, we discard the backward network.

The proposed method can be easily extended to the conditional generation case.

The forward hiddenstate transition is modified to h DISPLAYFORM1 where c denotes the task-dependent conditioning information, and similarly for the backward RNN.Bidirectional neural networks BID49 have been used as powerful feature extractors for sequence tasks.

The hidden state at each time step includes both information from the past and the future.

For this reason, they usually act as better feature extractors than the unidirectional counterpart and have been successfully used in a myriad of tasks, e.g. in machine translation , question answering BID10 and sequence labeling BID37 .

However, it is not straightforward to apply these models to sequence generation BID65 due to the fact that the ancestral sampling process is not allowed to look into the future.

In this paper, the backward model is used to regularize the hidden states of the forward model and thus is only used during training.

Both inference and sampling are strictly equivalent to the unidirectional case.

Gated architectures such as LSTMs BID27 and GRUs BID13 have been successful in easing the modeling of long term-dependencies: the gates indicate time-steps for which the network is allowed to keep new information in the memory or forget stored information.

BID20 ; Dieng et al. FORMULA1 ; BID18 effectively augment the memory of the network by means of an external memory.

Another solution for capturing long-term dependencies and avoiding gradient vanishing problems is equipping existing architectures with a hierarchical structure BID51 .

Other works tackled the vanishing gradient problem by making the recurrent dynamics unitary BID0 .

In parallel, inspired by recent advances in "learning to plan" for reinforcement learning BID52 BID55 , recent efforts try to augment RNNs with an explicit planning mechanism BID23 to force the network to commit to a plan while generating, or to make hidden states predictive of the far future BID34 .Regularization methods such as noise injection are also useful to shape the learning dynamics and overcome local correlations to take over the learning process.

One of the most popular methods for neural network regularization is dropout BID53 .

Dropout in RNNs has been proposed in BID41 , and was later extended in BID50 BID15 , where recurrent connections are dropped at random.

Zoneout BID32 modifies the hidden state to regularize the network by effectively creating an ensemble of different length recurrent networks.

BID31 introduce a "norm stabilization" regularization term that ensures that the consecutive hidden states of an RNN have similar Euclidean norm.

Recently, BID39 proposed a set of regularization methods that achieve state-of-the-art on the Penn Treebank language modeling dataset.

Other RNN regularization methods include the weight noise BID19 , gradient clipping BID45 and gradient noise BID42 .

We now present experiments on conditional and unconditional sequence generation, and analyze the results in an effort to understand the performance gains of TwinNet.

First, we examine conditional generation tasks such as speech recognition and image captioning, where the results show clear improvements over the baseline and other regularization methods.

Next, we explore unconditional language generation, where we find our model does not significantly improve on the baseline.

Finally, to further determine what tasks the model is well-suited to, we analyze a sequential imputation task, where we can vary the task from unconditional to strongly conditional.

We evaluated our approach on the conditional generation for character-level speech recognition, where the model is trained to convert the speech audio signal to the sequence of characters.

The forward and backward RNNs are trained as conditional generative models with softattention .

The context information c is an encoding of the audio sequence and the output sequence s is the corresponding character sequence.

We evaluate our model on the Wall Street Journal (WSJ) dataset closely following the setting described in BID4 .

We use 40 mel-filter bank features with delta and delta-deltas with their energies as the acoustic in-

We compare the attention model for speech recognition ("Baseline," BID4 ; the regularizer proposed by BID31 ("Stabilizing norm"); penalty on the L2 norm of the forward states BID39 ) ("AR"), which is equivalent to TwinNet when all the hidden states of the backward network are set to zero.

We report the results of our model ("TwinNet") both with g = I, the identity mapping, and with a learned g.

Test CER Valid CER Baseline 6.8 9.0 Baseline + Gaussian noise 6.9 9.1 Baseline + Stabilizing Norm 6.6 9.0 Baseline + AR 6.5 8.9 Baseline + TwinNet (g = I)6.6 8.7 Baseline + TwinNet (learnt g) 6.2 8.4puts to the model, these features are generated according to the Kaldi s5 recipe BID46 .

The resulting input feature dimension is 123.We observe the Character Error Rate (CER) for our validation set, and we early stop on the best CER observed so far.

We report CER for both our validation and test sets.

For all our models and the baseline, we follow the setup in BID4 and pretrain the model for 1 epoch, within this period, the context window is only allowed to move forward.

We then perform 10 epochs of training, where the context window looks freely along the time axis of the encoded sequence, we also perform annealing on the models with 2 different learning rates and 3 epochs for each annealing stage.

We use the AdaDelta optimizer for training.

We perform a small hyper-parameter search on the weight α of our twin loss, α ∈ {2.0, 1.5, 1.0, 0.5, 0.25, 0.1}, and select the best one according to the CER on the validation set.

Results We summarize our findings in TAB0 .

Our best performing model shows relative improvement of 12% comparing to the baseline.

We found that the TwinNet with a learned metric (learnt g) is more effective than strictly matching forward and hidden states.

In order to gain insights on whether the empirical usefulness comes from using a backward recurrent network, we propose two ablation tests.

For "Gaussian Noise," the backward states are randomly sampled from a Gaussian distribution, therefore the forward states are trained to predict white noise.

For "AR," the backward states are set to zero, which is equivalent to penalizing the norm of the forward hidden states BID39 .

Finally, we compare the model with the "Stabilizing Norm" regularizer BID31 , that penalizes the difference of the norm of consecutive forward hidden states.

Results shows that the information included in the backward states is indeed useful for obtaining a significant improvement.

Analysis The training/validation curve comparison for the baseline and our network is presented in FIG1 .

4 The TwinNet converges faster than the baseline and generalizes better.

The L2 cost raises in the beginning as the forward and backward network start to learn independently.

Later, due to the pressure of this cost, networks produce more aligned hidden representations.

FIG2 provides examples of utterances with L2 plotted along the time axis.

We observe that the high entropy words produce spikes in the loss for such words as "uzi."

This is the case for rare words which are hard to predict from the acoustic information.

To elaborate on this, we plot the L2 cost averaged over a word depending on the word frequency.

The average distance decreases with the increasing frequency.

The histogram comparison FIG1 ) for the cost of rare and frequent words reveal that the not only the average cost is lower for frequent words, but the variance is higher for rare words.

Additionally, we plot the dependency of the L2 cost cross-entropy cost of the forward network FIG1 ) to show that the conditioning also plays the role in the entropy of the output, the losses are not absolutely correlated.

We evaluate our model on the conditional generation task of image captioning task on Microsoft COCO dataset BID35 .

The MS COCO dataset covers 82,783 training images and 40,504 images for validation.

Due to the lack of standardized split of training, validation and test data, we follow Karpathy's split BID28 BID61 .

These are 80,000 training images and 5,000 images for validation and test.

We do early stopping based on the validation CIDEr scores and we report BLEU-1 to BLEU-4, CIDEr, and Meteor scores.

To evaluate the consistency of our method, we tested TwinNet on both encoder-decoder ('Show&Tell', BID59 and soft attention ('Show, Attend and Tell', BID61 image captioning models.

We use a Resnet BID25 with 101 and 152 layers pre-trained on ImageNet for image classification.

The last layer of the Resned is used to extract 2048 dimensional input features for the attention model BID61 .

We use an LSTM with 512 hidden units for both "Show & Tell" and soft attention.

Both models are trained with the Adam BID29 optimizer with a Table 2 : Results for image captioning on the MS COCO dataset, the higher the better for all metrics (BLEU 1 to 4, METEOR, and CIDEr).

We reimplement both Show&Tell BID59 and Soft Attention BID61 in order to add the twin cost.

We use two types of images features extracted either with Resnet-101 or Resnet-152.

DISPLAYFORM0 DeepVS BID28 62.5 45.0 32.1 23.0 19.5 66.0 ATT-FCN BID63 70.9 53.7 40.2 30.4 24.3 -Show & Tell BID59 ---27.7 23.7 85.5 Soft Attention BID61 70.7 49.2 34.4 24.3 23.9 -Hard Attention BID61 71.8 50.4 35.7 25.0 23.0 -MSM BID62 73.0 56.5 42.9 32.5 25.1 98.6 Adaptive Attention BID36 74 Table 3 : (left) Test set negative log-likelihood for binarized sequential MNIST, where denotes lower performance of our model with respect to the baselines.(right) Perplexity results on WikiText-2 and Penn Treebank BID39 .AWD-LSTM refers to the model of BID39 trained with the official implementation at http://github.com/salesforce/awd-lstm/.Model MNIST DBN 2hl BID16 ≈84.55 NADE BID57 88.33 EoNADE-5 2hl BID47 84.68 DLGM 8 BID48 ≈85.51 DARN 1hl ≈84.13 DRAW ≤80.97 P-Forcing (3-layer) BID33 79.58 PixelRNN(1-layer) BID44 80.75 PixelRNN(7-layer) BID44 79.20 PixelVAE BID24 79.02 MatNets BID1 78.50Baseline LSTM(3-layers) 79.87 + TwinNet(3-layers)

Baseline LSTM(3-layers) + dropout 79.59 + TwinNet(3-layers)79.12

LSTM BID64 82.2 78.4 4-layer LSTM BID38 67.9 65.4 5-layer RHN BID38 64 BID38 78.1 75.6 1-layer LSTM BID38 69.3 65.9 2-layer LSTM BID38 69.1 65.9AWD-LSTM 68.7 65.8 + TwinNet 68.0 64.9 learning rate of 10 −4 .

TwinNet showed consistent improvements over "Show & Tell" (Table 2 ).

For the soft attention model we observe small but consistent improvements for majority of scores.

We investigate the performance of our model in pixel-by-pixel generation for sequential MNIST.

We follow the setting described by BID33 : we use an LSTM with 3-layers of 512 hidden units for both forward and backward LSTMs, batch size 20, learning rate 0.001 and clip the gradient norms to 5.

We use Adam BID29 as our optimization algorithm and we decay the learning rate by half after 5, 10, and 15 epochs.

Our results are reported at the Table 3 (left).

Our baseline LSTM implementation achieves 79.87 nats on the test set.

We observe that by adding the TwinNet regularization cost consistently improves performance in this setting by about 0.52 nats.

Adding dropout to the baseline LSTM is beneficial.

Further gains were observed by adding both dropout and the TwinNet regularization cost.

This last model achieves 79.12 nats on test set.

Note that this result is competitive with deeper models such as PixelRNN (Oord et al., 2016b) (7-layers) and PixelVAE BID24 which uses an autoregressive decoder coupled with a deep stochastic auto-encoder.

As a last experiment, we report results obtained on a language modelling task using the PennTree Bank and WikiText-2 datasets BID39 .

We augment the state-of-the-art AWD-LSTM model BID39 with the proposed TwinNet regularization cost.

The results are reported in Table 3 (right).

In this paper, we presented a simple recurrent neural network model that has two separate networks running in opposite directions during training.

Our model is motivated by the fact that states of the forward model should be predictive of the entire future sequence.

This may be hard to obtain by optimizing one-step ahead predictions.

The backward path is discarded during the sampling and evaluation process, which makes the sampling process efficient.

Empirical results show that the proposed method performs well on conditional generation for several tasks.

The analysis reveals an interpretable behaviour of the proposed loss.

One of the shortcomings of the proposed approach is that the training process doubles the computation needed for the baseline (due to the backward network training).

However, since the backward network is discarded during sampling, the sampling or inference process has the exact same computation steps as the baseline.

This makes our approach applicable to models that requires expensive sampling steps, such as PixelRNNs BID44 and WaveNet (Oord et al., 2016a) .

One of future work directions is to test whether it could help in conditional speech synthesis using WaveNet.

We observed that the proposed approach yield minor improvements when applied to language modelling with PennTree bank.

We hypothesize that this may be linked to the amount of entropy of the target distribution.

In these high-entropy cases, at any time-step in the sequence, the distribution of backward states may be highly multi-modal (many possible futures may be equally likely for the same past).

One way of overcoming this problem would be to replace the proposed L2 loss (which implicitly assumes a unimodal distribution of the backward states) by a more expressive loss obtained by either employing an inference network BID30 or distribution matching techniques BID17 .

We leave that for future investigation.

@highlight

The paper introduces a method of training generative recurrent networks that helps to plan ahead. We run a second RNN in a reverse direction and make a soft constraint between cotemporal forward and backward states.