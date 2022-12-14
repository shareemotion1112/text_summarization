In this paper, we introduce a novel method to interpret recurrent neural networks (RNNs), particularly long short-term memory networks (LSTMs) at the cellular level.

We propose a systematic pipeline for interpreting individual hidden state dynamics within the network using response characterization methods.

The ranked contribution of individual cells to the network's output is computed by analyzing a set of interpretable metrics of their decoupled step and sinusoidal responses.

As a result, our method is able to uniquely identify neurons with insightful dynamics, quantify relationships between dynamical properties and test accuracy through ablation analysis, and interpret the impact of network capacity on a network's dynamical distribution.

Finally, we demonstrate generalizability and scalability of our method by evaluating a series of different benchmark sequential datasets.

expose them to defined input signals such as step and sinusoid functions.

Through evaluation of output attributes, such as response settling time, phase-shift, and amplitude, we demonstrate that it is possible to predict sub-regions of the network dynamics, rank cells based on their relative contribution to network output, and thus produce reproducible metrics of network interpretability.

For example, step response settling time delineates cells with fast and slow response dynamics.

In addition, by considering the steady-state value of the cellular step response and the amplitude of the sinusoid response, we are able to identify cells that significantly contribute to a network's decision.

We evaluate our methodology on a range of sequential datasets and demonstrate that our algorithms scale to large LSTM networks with millions of parameters.

The key contributions of this paper can be summarized as follows:1.

Design and implementation of a novel and lightweight algorithm for systematic LSTM interpretation based on response characterization;2.

Evaluation of our interpretation method on four sequential datasets including classification and regression tasks; and 3.

Detailed interpretation of our trained LSTMs on the single cell scale via distribution and ablation analysis as well as on the network scale via network capacity analysis.

First, we discuss related work in Sec. 2 and then introduce the notion of RNNs as dynamic systems in Sec. 3.

Sec. 4 presents our algorithm for response characterization and defines the extracted interpretable definitions.

Finally, we discuss the interpretations enabled by this analysis in Sec. 5 through an series of experiments, and provide final conclusions of this paper in Sec. 6.

Deep Neural Networks Interpretability -A number of impactful attempts have been proposed for interpretation of deep networks through feature visualization BID9 BID37 BID36 BID20 BID34 BID6 .

Feature maps can be empirically interpreted at various scales using neural activation analysis BID27 , where the activations of hidden neurons or the hidden-state of these neurons is computed and visualized.

Additional approaches try to understand feature maps by evaluating attributions BID32 BID10 BID21 BID35 .

Feature attribution is commonly performed by computing saliency maps (a linear/non-linear heatmap that quantifies the contribution of every input feature to the final output decision).

The contributions of hidden neurons, depending on the desired level of interpretability, can be highlighted at various scales ranging from individual cell level, to the channel and spatial filter space, or even to arbitrary groups of specific neurons BID27 .

A dimensionality reduction method can also be used to abstract from high dimensional feature maps into a low dimensional latent space representation to qualitatively interpret the most important underlying features BID25 BID5 .

However, these methods often come with the cost of decreasing cell-level auditability.

Richer infrastructures have been recently developed to reason about the network's intrinsic kinetics.

LSTMVis BID34 , relates the hidden state dynamics patterns of the LSTM networks to similar patterns observed in larger networks to explain an individual cell's functionality.

A systematic framework has also been introduced that combines interpretability methodologies across multiple network scales BID27 .

This enables exploration over various levels of interpretability for deep NNs; however, there is still space to incorporate more techniques, such as robust statistics BID22 , information theory approaches BID31 , gradients in correlation-domain BID14 and response characterization methods which we address in this paper.

Recurrent Neural Networks Interpretability -Visualization of the hidden-state of a fixed-structure RNNs on text and linguistic datasets identifies interpretable cells which have learned to detect certain language syntaxes and semantics BID20 BID34 .

RNNs have also been shown to learn input-sensitive grammatical functions when their hidden activation patterns were visualized BID18 BID19 .

Moreover, gradient-based attribution evaluation methods were used to understand the RNN functionality in localizing key words in the text.

While these techniques provide rich insight into the dynamics of learned linguistics networks, the interpretation of the network often requires detailed prior knowledge about the data content.

Therefore, such methods may face difficulties in terms of generalization to other forms of sequential data such as time-series which we focus on in our study.

Another way to build interpretability for RNNs is using the attention mechanism where the network architecture is constrained to attend to a particular parts of the input space.

RNNs equipped with an attention mechanism have been successfully applied in image captioning, the fine-alignments in machine translation, and text extraction from documents BID15 .

Hidden-state visualization is a frequently shared property of all of these approaches in order to effectively understand the internals of the network.

Hudson et al. BID17 also introduced Memory, Attention, and Composition (MAC) cells which can be used to design interpretable machine reasoning engines in an end-to-end fashion.

MAC is able to perform highly accurate reasoning, iteratively directly from the data.

However, application of these modification to arbitrary network architectures is not always possible, and in the case of LSTM specifically, the extension is not possible in the current scope of MAC.Recurrent Neural Networks Dynamics-Rigorous studies of the dynamical systems properties of RNNs, such as their activation function's independence property (IP) BID4 , state distinguishability BID2 , and observability BID11 BID12 date back to more than two decades.

Thorough analyses of how the long term dynamics are learned by the LSTM networks has been conducted in BID16 .

Gate ablation analysis on the LSTM networks has been performed to understand cell's dynamics BID13 BID8 .

We introduce the response characterization method, as a novel building block to understand and reason about LSTM hidden state dynamics.

In this section, we briefly we recap kinetics of RNNs.

We denote the global dynamics of the hidden state values as h l t , with t ??? {1..T } denoting the time, and l ??? {1..L} representing the layers of the neural network.

A vanilla recurrent neural network (RNN) can be formulated as BID29 BID20 : DISPLAYFORM0 where DISPLAYFORM1 shows the weight matrix.

h 0 t retains an input vector x t and h L t holds a vector at the last hidden layer, L, that is mapped to an output vector y t which is ultimately the function of all input sequence {x 1 , . . .

, x T }.RNNs are formulated as control dynamical systems in the form of the following differential equation (For the sake of notation simplicity, we omit the time argument, t): DISPLAYFORM2 where h denotes its internal state ( ?? illustrates time-shift or time derivative for the discrete and continuous-time systems, respectively), x stands for the input, and R [n??n] , W [n??m] and C [p??n] are real matrices representing recurrent weights, input weights and the output gains, respectively.

?? : R ??? R indicates the activation function.

In the continuous setting, ?? should be locally Lipschitz (see BID3 for a more detailed discussion).

Long short term Memory (LSTM) BID16 , are gated-recurrent neural networks architectures specifically designed to tackle the training challenges of RNNs.

In addition to memorizing the state representation, they realize three gating mechanisms to read from input (i), write to output (o) and forget what the cell has stored (f ).

Activity of the cell can be formulated as follows BID13 : rise to distinct outputs for two different initial states at which the system is started BID33 .

Observable systems realize unique internal parameter settings BID4 .

One can then reason about that parameter setting to interpret the network for a particular input profile.

Information flow in LSTM networks carries on by the composition of static and time-varying dynamical behavior.

This interleaving of building blocks makes a complex partially-dependent sets of nonlinear dynamics that are hard to analytically formulate and to verify their observability properties As an alternative, in this paper we propose a technique for finding sub-regions of hidden observable dynamics within the network with a quantitative and systematic approach by using response characterization.

DISPLAYFORM0 DISPLAYFORM1

In this section, we explore how response characterization techniques can be utilized to perform systematic, quantitative, and interpretable understanding of LSTM networks on both a macro-network and micro-cell scale.

By observing the output of the system when fed various baseline inputs, we enable a computational pipeline for reasoning about the dynamics of these hidden units.

Figure 1 provides a schematic for our response characterization pipeline.

From a trained LSTM network, comprising of M LSTM units, we isolate individual LSTM cells, and characterize their output responses based on a series of interpretable response metrics.

We formalize the method as follows:Definition 1 Let G, be a trained LSTM network with M hidden LSTM units.

Given the dynamics of the training dataset (number of input/output channels, the main frequency components, the amplitude range of the inputs), we design specific step and sinusoidal inputs to the network, and get the following insights about the dynamics of the network at multi-scale resolutions:??? the relative strength or contribution of components within the network;??? the reactiveness of components to sudden changes in input; and??? the phase alignment of the hidden outputs with respect to the input.

Specifically, we analyze the responses of (1) the step input and (2) the sinusoidal input.

We use the classic formulations for each of these input signals wherein (1) step: Across a network of LSTM units we can approximate sub-regions of the dynamics of a single cell, u, by extracting the input and recurrent weights corresponding to that individual cell.

We then define a sub-system consisting of just that single cell and subsequently feed one of our baseline input signals, x t ??? t???{1..T } to observe the corresponding output response, y t .

In the following, we define the interpretable response metrics for the given basis input used in this study: DISPLAYFORM0 Definition 2 The initial and final response of the step response signal is the starting and steady state responses of the system respectively, while the response output change represents their relative difference.

Response output change or the delta response for short determines the strength of the LSTM unit with a particular parameter setting, in terms of output amplitude.

This metric can presumably detect significant contributor units to the network's decision.

The settling time of the step response is elapsed time from the instantaneous input change to when the output lies within a 90% threshold window of its final response.

Computing the settling time for individual LSTM units enables us to discover "fast units" and "slow units".

This leads to the prediction of active cells when responding to a particular input profile.

The amplitude and frequency of a cyclic response signal is the difference in output and rate at which the response output periodically cycles.

The response frequency,f , is computed by evaluating the highest energy component of the power spectral density:f = arg max S yy (f ).The amplitude metric enables us to rank LSTM cells in terms of significant contributions to the output.

This criteria is specifically effective in case of trained RNNs on datasets with a cyclic nature.

Given a sinusoidal input, phase-shifts and phase variations expressed at the unit's output, can be captured by evaluating the frequency attribute.

The correlation of the output response with respect to the input signal is the dot product between the unbiased signals: DISPLAYFORM0 The correlation metric correspondes to the phase-alignments between input and output of the LSTM unit.

Systematic computation of each of the above responses metrics for a given LSTM dynamics, enables reasoning on the internal kinetics of that system.

Specifically, a given LSTM network can be decomposed into its individual cell components, thus creating many smaller dynamical systems, which can be analyzed according to their individual response characterization metrics.

Repeating this process for each of the cells in the entire network creates two scales of dynamic interpretability.

Firstly, on the individual cell level within the network to identify those which are inherently exhibiting fast vs slow responses to their input, quantify their relative contribution towards the system as a whole, and even interpret their underlying phase-shift and alignment properties.

Secondly, in addition to characterizing responses on the cell level we also analyze the effect of network capacity on the dynamics of the network as a whole.

Interpreting hidden model dynamics is not only interesting as a deployment tool but also as a debugging tool to pinpoint possible sources of undesired dynamics within the network.

While one can use these response characterization techniques to interpret individual cell dynamics, this analysis can also be done on the aggregate network scale.

After computing our response metrics for all decoupled cells independently we then build full distributions over the set of all individual pieces of the network to gain understanding of the dynamics of the network as a whole.

This study of the response metric distributions presents another rich representation for reasoning about the dynamics, no longer at a local cellular scale, but now, on the global network scale.

In the following section, we provide concrete results of our system in practice to interpret the dynamics of trained LSTMs for various sequence modeling tasks.

We present our computed metric response characteristics both on the decoupled cellular level as well as the network scale, and provide detailed and interpretable reasoning for these observed dynamics.

We chose four benchmark sequential datasets and trained on various sized LSTM networks ranging from 32 to 320 LSTM cell networks.

The results and analysis presented in this section demonstrate applicability of our algorithms to a wide range of temporal sequence problems and scalability towards deeper network structures.

We start by reasoning how our response characterization method can explain the hidden-state dynamics of learned LSTM networks for a sequential MNIST dataset and extend our findings to three additional datasets.

We perform an ablation analysis and demonstrate how some of our metrics find cells with significant contributions to the network's decision, across all datasets.

We start by training an LSTM network with 64 hidden LSTM cells to classify a sequential MNIST dataset.

Inputs to the cells are sequences of length 784 generated by stacking the pixels of the 28 ?? 28 hand-writing digits, row-wise (cf.

FIG1 ) and the output is the digit classification.

Individual LSTM cells were then isolated and their step and sine-response were computed for the attributes defined formerly (cf.

Fig. 4 ).

FIG1 ).

This interpretation allows us to indicate fast-activated/deactivated neurons at fast and slow phases of a particular input sequence.

This is validated in FIG1 , where the output state of individual LSTM cells are visually demonstrated when the network receives a sequence of the digit 6.

The figure is sorted in respect to the predicted settling time distribution.

We observe that fast-cells react to fast-input dynamics almost immediately while slow-cells act in a slightly later phase.

This effect becomes clear as you move down the heatmap in FIG1 and observe the time difference from the original activation.

The distribution of the delta-response, indicates inhibitory and excitatory dynamics expressed by a 50% ratio (see FIG1 ).

This is confirmed by the input-output correlation criteria, where almost half of the neurons express antagonistic behavior to their respective sine-wave input FIG1 .

The sine-frequency distribution depicts that almost 90% of the LSTM cells kept the phase, nearly aligned to their respective sine-input, which indicates existence of a linear transformation.

A few cells learned to establish a faster frequencies than their inputs, thereby realizing phase-shifting dynamics FIG1 .

The sine-amplitude distribution in FIG1 demonstrates that the learned LSTM cells realized various amplitudes that are almost linearly increasing.

The ones with a high amplitude can be interpreted as those maximally contributing to the network's decision.

In the following sections, we investigate the generalization of these effects to other datasets.

We trained LSTM networks with 128 hidden cells, for four different temporal datasets: sequential MNIST BID23 , S&P 500 stock prices BID1 and CO 2 concentration for the Mauna Laua volcano BID0 forecasting, and classification of protein secondary structure BID30 .

Learned networks for each dataset are denoted seq-MNIST, Stock-Net, CO 2 -Net and Protein-Net.

all five metrics with the network size of 128.

It represents the average cell response metric attributes for various datasets and demonstrates the global speed and amplitude of the activity of network in terms of dynamical properties of the response characterization metrics.

Fig 3A- E, represents the distributions for the metrics sorted by the value of their specific attribute across all datasets.

Cells in Protein-Net realized the fastest dynamics (i.e. smallest settling time) compared to the other networks, while realizing a similar trend to the seq-MNIST (Fig. 3A) .

The settling time distribution for the LSTM units of CO 2 and Stock-Net depicts cell-groups with similar speed profiles.

For instance neurons 52 to 70 in Stock-Net, share the same settling time (Fig. 3A) .

Sine frequency stays constant for all networks except from some outliers which tend to modify their input-frequency (Fig. 3D) .

The delta response and the correlation metrics ( Fig. 3B and Fig. 3E ) both indicate the distribution of the inhibitory and excitatory behavior of individual cells within the network.

Except from the Seq-MNIST net, neurons in all networks approximately keep a rate of 44% excitatory and 56% inhibitory dynamics.

The high absolute amplitude neurons (at the two tails of Fig. 3C ), are foreseen as the significant contributors to the output's decision.

We validate this with an ablation analysis subsequently.

Moreover, most neurons realize a low absolute delta-response value, for all datasets except for MNIST (Fig. 3B) .

This is an indication for cells with an equivalent influence on the output accuracy.

Sine-amplitude stays invariant for most neurons in Stock and CO 2 -Nets (Fig. 3C ).

For the seq-MNIST net and Protein-net, this distribution has a gradually increasing trend with weak values.

This predicts that individual cells are globally equivalently contributing to the output.

To assess the quality of the predictions and interpretations of the provided response characterization metrics, we performed individual cell-ablation analysis on LSTM networks and evaluated the cellimpact on the output accuracy (misclassification rate), for the classification problems and on the output performance (mean absolute error), for the regression problems.

We knocked out neurons from trained LSTM networks with 128 neurons.

Fig. 4A -H illustrate the performance of the network for individual cell ablations for all four datasets.

The gray solid line in each subplot, stands for the predictions of the response metrics.

For CO 2 -Net, this confirms that neurons with higher sine amplitude tend to disrupt the output more (Fig 4D) .

For the same network, the delta response predicted that neurons with high negative or positive value, are more significant in output's prediction.

This is clearly illustrated in Fig. 4C .

For seq-MNIST-Net, the same conclusions held true; neurons with high absolute value of delta response or sine-amplitude reduce the accuracy at the output dramatically (Fig. 4A-B) .

By analyzing the sine-amplitude and delta-response of Protein-Net, we observe that neurons are equivalently valued and tend to contribute equivalently to the output accuracy.

This is verified in the ablation analysis, shown in Fig. 4G and 4H , where the mean-misclassification error rate stays constant for all neural ablations.

The absolute value for Stock-Net was also weak in terms of these two metrics, though there were some outliers at the tails of their distribution that predicted dominant neurons.

This is clearly notable when comparing the neurons 120 to 128 of Fig. 4F to their prediction (gray line) where the amplitude of the response is maximal.

In Fig. 4E ablation experiments for neurons 1 to 40 and 100 to 128 impose higher impact on the overall output.

This was also observed in the delta response prediction shown in 4B, since neurons with stronger output response were present at the two tails of the distribution.

While we analyzed the response characterization distributions on a cellular level above, in this subsection we focus on the effect of network capacity on observed hidden dynamics of the system on a global scale.

Reasoning on this scale allows us to draw conclusions on how increasing the expressive capacity of LSTM networks trained on the same dataset can result in vastly different learned dynamics.

We experimentally vary the capacity by simply adding hidden LSTM cells to our network and retraining on the respective dataset from scratch.

The relationship between each response characteristic metric and the network capacity is visualized in FIG4 -E. The trends across datasets are visualized in a single subplot to compare respective trends.

One especially interesting result of this analysis is the capacity relationship with response amplitude (cf.

FIG4 ).

Here we can see that the amplitude response decays roughly proportionally to 1 N , for all datasets, where N is the number of LSTM cells.

In other words, we get the intuitive finding that as we increase the number of LSTM cells, the magnitude of each cell's relative contribution needed to make a prediction will subsequently decrease.

Yet another key finding of this analysis is that the distribution of settling time is relatively constant across network capacity (cf.

FIG4 ).

Intuitively, this means that the network is able to learn the underlying time delay constants represented in the dataset irrespective of the network capacity.

One particularly interesting point comes for Protein-Net which exhibits vastly different behavior for both settling time FIG4 ) than the remainder of the datasets.

Upon closer inspection, we found that Protein-Net was heavily overfitting with increased capacity.

This can be seen as an explanation for the rapid decay in its settling time as the addition of LSTM cells would increase specificity of particular cells and exhibit dynamical properties aligning with effectively memorizing pieces of the training set.

In this paper, we proposed a method for response characterization for LSTM networks to predict cell-contributions to the overall decision of a learned network on both the cell and network-level resolution.

We further verified and validated our predictions by performing an ablation analysis to identify cell's which contribution heavily to the network's output decision with our simple response characterization method.

The resulting method establishes a novel building block for interpreting LSTM networks.

The LSTM network's dynamic-space is broad and cannot be fully captured by fundamental input sequences.

However, our methodology demonstrates that practical sub-regions of dynamics are reachable by response metrics which we use to build a systematic testbench for LSTM interpretability.

We have open-sourced our algorithm to encourage other researchers to further explore dynamics of LSTM cells and interpret the kinetics of their sequential models.

In the future, we aim to extend our approach to even more data modalities and analyze the training phase of LSTMs to interpret the learning of the converged dynamics presented in this work.7 Acknowledgment

@highlight

Introducing the response charactrization method for interpreting cell dynamics in learned long short-term memory (LSTM) networks. 