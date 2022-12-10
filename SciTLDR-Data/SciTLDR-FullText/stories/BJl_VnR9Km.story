In this paper we developed a hierarchical network model, called Hierarchical Prediction Network (HPNet) to understand how spatiotemporal memories might be learned and encoded in a representational hierarchy for predicting future video frames.

The model is inspired by the feedforward, feedback and lateral recurrent circuits in the mammalian hierarchical visual system.

It assumes that spatiotemporal memories are encoded in the recurrent connections within each level and between different levels of the hierarchy.

The model contains a feed-forward path that  computes and encodes spatiotemporal features of successive complexity and a feedback path that projects interpretation from a higher level to the level below.

Within each level, the feed-forward path and the feedback path intersect in a recurrent gated circuit that integrates their signals as well as the circuit's internal memory states to generate  a prediction of the incoming signals.

The network learns by comparing the incoming signals with its prediction, updating its internal model of the world by minimizing the prediction errors at each level of the hierarchy in the style of {\em predictive self-supervised learning}. The network processes data in blocks of video  frames rather than  a frame-to-frame basis.

This allows it to learn relationships among movement patterns, yielding state-of-the-art performance in long range video sequence predictions in benchmark datasets.

We observed that hierarchical interaction in the network introduces sensitivity to memories of global movement patterns even in the population representation of the units in the earliest level.

Finally, we provided neurophysiological evidence, showing that neurons in the early visual cortex of awake monkeys exhibit very similar sensitivity and behaviors.

These findings suggest that  predictive self-supervised learning might be an important principle for representational learning in the visual cortex.

While the hippocampus is known to play a critical role in encoding episodic memories, the storage of these memories might ultimately rest in the sensory areas of the neocortex BID27 .

Indeed, a number of neurophysiological studies suggest that neurons throughout the hierarchical visual cortex, including those in the early visual areas such as V1 and V2, might be encoding memories of object images and of visual sequences in cell assemblies BID54 BID14 BID52 BID2 BID21 .

As specific priors, these memories, together with the generic statistical priors encoded in receptive fields and connectivity of neurons, serve as internal models of the world for predicting incoming visual experiences.

In fact, learning to predict incoming visual signals has also been proposed as an objective that drives representation learning in a recurrent neural network in a self-supervised learning paradigm, where the discrepancy between the model's prediction and the incoming signals can be used to train the network using backpropagation, without the need of labeled data BID9 BID46 BID42 BID34 BID21 .In computer vision, a number of hierarchical recurrent neural network models, notably PredNet BID24 and PredRNN++ , have been developed for video prediction with state-of-the-art performance.

PredNet, in particular, was inspired by the neuroscience principle of predictive coding BID31 BID39 BID21 BID7 BID11 .

It learns a LSTM (long short-term memory) model at each level to predict the prediction errors made in an earlier level of the hierarchical visual system.

Because the error representations are sparse, the computation of PredNet is very efficient.

However, the model builds a hierarchical representation to model and predict its own errors, rather than learning a hierarchy of features of successive complexities and scales to model the world.

The lack of a compositional feature hierarchy hampers its ability in long range video predictions.

Here, we proposed an alternative hierarchical network architecture.

The proposed model, HPNet (Hierarchical Prediction Network), contains a fast feedforward path, instantiated currently by a fast deep convolutional neural network (DCNN) that learns a representational hierarchy of features of successive complexity, and a feedback path that brings a higher order interpretation to influence the computation a level below.

The two paths intersect at each level through a gated recurrent circuit to generate a hypothetical interpretation of the current state of the world and make a prediction to explain the bottom-up input.

The gated recurrent circuit, currently implemented in the form of LSTM, performs this prediction by integrating top-down, bottom-up, and horizontal information.

The discrepancy between this prediction and the bottom-up input at each level is called prediction error, which is fed back to influence the interpretation of the gated recurrent circuits at the same level as well as the level above.

To facilitate the learning of relationships between movement patterns, HPNet processes data in the unit of a spatiotemporal block that is composed of a sequence of video frames, rather than frame by frame, as in PredNet and PredRNN++.

We used a 3D convolutional LSTM at each level of the hierarchy to process these spatiotemporal blocks of signals BID1 , which is a key factor underlying HPNet's better performance in long range video prediction.

In the paper, we will first demonstrate HPNet's effectiveness in predictive learning and its competency in long range video prediction.

Then we will provide neurophysiological evidence showing that neurons in the early visual cortex of the primate visual system exhibit the same sensitivity to memories of global movement patterns as units in the lowest modules of HPNet.

Our results suggest that predictive self-supervised learning might indeed be an important strategy for representation learning in the visual cortex, and that HPNet is a viable computational model for understanding the computation in the visual cortical circuits.

Our objective is to develop a hierarchical cortical model for predictive learning of spatiotemporal memories that is competitive both for video prediction, and for understanding the learning principles and the computational mechanisms of the hierarchical visual system.

In this regard, our model is similar conceptually to Ullman's counter-stream model (Ullman, 1995), Mumford's analysis by synthesis framework BID32 , and Hawkin's hierarchical spatiotemporal memory model (HTM) BID15 for hierarchical cortical processing.

At a conceptual level, it can also be considered as a deep learning implementation of hierarchical Bayesian inference model of the visual cortex BID22 BID5 BID18 .HPNet integrates ideas of predictive coding BID32 BID39 BID24 and associative coding BID28 BID13 .

It differs from the predictive coding models BID39 BID24 in that it learns a hierarchy of feature representations in the feedforward path to model features in the world as in normal deep convolutional neural networks (DCNN).

PredNet, on the other hand, builds a hierarchy to model successive prediction errors of its own prediction of the world.

PredNet is efficient because its convolution is operated on sparse prediction error codes, but we believe lacking a hierarchical feature representation limits its ability to model relationships among more global and abstract movement concepts for longer range video prediction.

We believe having a fast bottom-up hierarchy of spatiotemporal features of successive scale and abstraction will allow the system to see further into the future and make better prediction.

A key difference between the genre of predictive learning models (HPNet, PredNet) and the earlier predictive coding models implemented by Kalman filters BID39 or associative coding models implemented by interactive activation BID28 BID13 is that the synthesis of expectation is not done simply by the feedback path, via weight matrix multiplication, but by local gated recurrent circuits at each level.

This key feature makes this genre of predictive learning models more powerful and competent in solving real computer vision problems.

The idea of predictive learning, using incoming video frames as self-supervising teaching labels to train recurrent networks, can be traced back to BID9 .

Recently, there has been active exploration of self-supervised learning in computer vision BID35 BID34 BID12 BID42 BID37 BID47 , particularly in the area of video prediction research BID17 BID43 BID53 BID33 BID46 BID51 .

The large variety of models can be roughly grouped into three categories: autoencoders, DCNN, and hierarchy of LSTMs.

Some models also involve feedforward and feedback paths, where the feedback paths have been implemented by deconvolution, autoencoder networks, LSTM or adversary networks BID10 BID24 BID48 BID11 .

Some other models, such as variational autoencoders, allowed multiple hypotheses to be sampled BID0 BID6 .PredRNN++ is the state-of-the-art hierarchical model for video prediction.

It consists of a stack of LSTM, with the LSTM at one level providing feedforward input directly to the LSTM at the next level, and ultimately predicting the next video frame at its top level.

Thus, its hierarchical representation is more similar to an autoencoder, with the intermediate layers modeling the most abstract and global spatiotemporal memories of movement patterns and the subsequent layers representing the unfolding of the feedback path into a feedforward network with its top-layer's output providing the prediction of the next frame.

PredRNN++ does not claim neural plausibility, but it offers state-of-the-art performance for benchmark performance evaluation, with documented comparisons to other approaches.

Recent single-unit recording experiments in the inferotemporal cortex (IT) of monkeys have shown that neurons responded significantly less to predictable sequences than to novel sequences BID29 BID30 BID38 , suggesting that neural activities might signal prediction errors.

The novel neurophysiolgical experiment we presented here demonstrated similar prediction suppression effects in the early visual cortex of monkeys for well-learned videos, suggesting neuronal sensitivity to memories of global movement patterns and scene context in the earliest visual areas.

This is consistent with other recent studies that showed neurons in mouse V1 might be able to encode some forms of spatiotemporal memories in their recurrent circuits BID14 BID52 BID3 .

HPNet is composed of a stack of Cortical Modules (CM).

Each CM can be considered as a visual area along the ventral stream of the primate visual system, such as V1, V2, V4 and IT.

We used four Cortical Modules in our experiment.

The network contains a feedforward path that is realized in a deep convolutional neural network (DCNN), a stack of Long Short Term Memory (LSTM) modules that link the feedforward path and the feedback path together.

Figure 1 (a) shows two CMs stacked on top of each other.

The feedforward path performs convolution (indicated by ) on the input spatiotemporal block I l with a kernel to produce R l , where l indicates CM level.

R l is then down-sampled to provide the input I l+1 for CM l+1 for another round of convolution in the feedforward path.

I l+1 also goes into LSTM l+1 (Lhe STM in CMt l+1 ).

In each CM l level, the bottom-up input I l is compared with the prediction P l generated from the interpretation output H l of LSTM l .

The prediction error signal is transformed by a convolution into E l , which is fed back to both LSTM l and LSTM l+1 to influence their generation of new hypotheses H l and H l+1 .

To make the timing relationship between the different interacting variables more explicit, we now use k to indicate time step or, equivalently, the video input frame.

LSTM l at step k integrates the bottom-up feature input R to recover the representation at the current frame R k l .

This allows the network to maintain a full higher order representation R at all times in the next layer while enjoying the benefit of fast computation on sparse input.

In their scheme BID36 , the first frame I k=0 was convolved with a set of dense convolution kernels and then the subsequent frames were convolved with a set of sparse convolution kernels.

For parsimony and neural plausibility, we used the same set of sparse kernels for processing both the first full frame and the subsequent temporal-difference frames, at the expense of incurring some inaccuracy in our prediction of the first few frames.

The input data of our network model is a sequence of video frames or a spatiotemporal block.

For our implementation, each block contains 5 video frames.

If we consider that each frame corresponds roughly to 25 ms, this would translate into 125 ms, in the range of the length of temporal kernel of a cortical neuron.

Our convolution kernel is in three dimension, processing the video by spatiotemporal blocks.

The block could slide in time with a temporal stride of one frame or a stride as large as the length of the block d. The LSTM is a 3D convolutional LSTM BID1 because of 3D convolution and spatiotemporal blocks.

Convolution LSTM BID41 , in which Hadamard product in LSTM is replaced by a convolution, has greatly improved the performance of LSTM in many applications.

Earlier video prediction models (e.g. PredNet, PredRNN) processed video sequences frame by frame, as shown in FIG1 (d).

We experimented with different data units and approaches.

In the Frame-to-Frame (F-F) approach, an input frame is used to generate one predicted future frame FIG1 ).

In the Block-to-Frame (B-F) approach FIG1 ), a block of input frames is used to generate one predicted future frame.

This approach is time consuming, but provides more accurate near-range predictions.

For longer-range predictions, we found using a spatiotemporal block to predict a spatiotemporal block, i.e. the Block-to-Block (B-B) approach FIG1 ), to be the most effective, because the LSTM learns the relationship between movement segments in the sequences.

The details of our algorithm of the 3D convolutional LSTM is specified in Appendix A.

The entire network is trained by minimizing a loss function which is the weighted sum of all the prediction errors, with the following algorithm, DISPLAYFORM0 DISPLAYFORM1 where x t is the input sequence, H k l is the output of LSTM, P k l is the prediction, SATLU is a saturating non-linearity set at the maximum pixel value: SATLU(x; p max ):= min(p max , x), spconv is sparse convolution, λ k and λ l are weighting factors by time and CM level, respectively, and n l is the number of units in the lth CM level, and d is the number of frames in each spatiotemporal block.

The full algorithm is shown in Algorithm 1.

In this section, we first evaluate the performance of our model in video prediction using two benchmark datasets: (1) synthetic sequences of the Moving-MNIST database and (2) the KTH 1 real world human movement database.

We then investigate the representations in the model to understand how recurrent network structures have impacted on the feedforward representation.

We finally compare the temporal activities of neurons in the network model with that of neurons in the visual cortex of monkeys, in video sequence learning, to evaluate the plausibility of HPNet.

Since for video prediction, PredNet is the most neurally plausible model and PredRNN++ provides state-of-the-art computer vision performance, we will compare HPNet's performance with these two network models.

Because these two models work on frame-to-frame basis, we implemented three versions of our network for comparison: (1) Frame-to-Frame (F-F), where we set our data spatiotemporal block size to one frame and used 2D convLSTM instead of 3D convLSTM to predict the next frame based on the current frame; (2) Block-to-Frame (B-F), where we used a sliding block window to predict the next frame based on the current block of frames; (3) Block-to-Block (B-B), where the next spatiotemporal block was predicted from the current spatiotemporal block FIG1 ).We trained all five networks using 40-frame sequences extracted from the two databases in the same way as described in BID24 .

We then compared their performance in predicting the next 20 frames when only the first 20 frames were given.

The test sequences were drawn from the same dataset but not in the training set.

The common practice in PreNet and PredRNN++ for predicting future frames when input is no longer available is to make the prediction of the last time step the next input and use that to generate prediction of the next time step.

All models tested have four modules (layers).

All three versions of our model and PredNet used the same number of feature channels in each layer, optimized by grid search, i.e. (16, 32, 64, 128) for the and (24, 48, 96, 192) for the KTH dataset.

For PredRNN++, we used Algorithm 1 The algorithm of our model DISPLAYFORM0 end if 8: end for 9:for l = 1 to L do Bottom-up procedure 10:if l = 1 then 11: DISPLAYFORM1 end for 18: end for the same architecture and feature channel numbers provided by .

All kernel sizes are either 3×3 (for F-F) or 3×3×3 (for B-F and B-B) for all five models.

The input image frame's spatial resolution is 64×64.The models were trained and tested on GeForce GTX TITAN X GPUs.

We evaluated the prediction performance based on two quantitative metrics: Mean-Squared Error (MSE) and the standard Structural Similarity Index Measure (SSIM) BID50 of the last 20 frames between the predicted frames and the actual frames.

The values of SSIM range from -1 to 1, with a larger value indicating greater similarity between the predicted frames and the actual future frames.

We randomly chose subsets of digits in the Moving MNIST 2 dataset in which the video sequences contain two handwritten digits bouncing inside a frame of 64×64 pixels.

We extracted 40-frame sequences at random starting frame position in the video in the same way as in BID42 (followed by PredNet and PredRNN++).

This extraction process is repeated 15000 times, resulting in a training set of 10000 sequences, a validation set of 2000 sequences, and a testing set of 3000 sequences.

FIG2 and TAB0 compare the results of different models on the Moving-MNIST dataset.

There are 40 frames in total and we show the results every two frames.

The yellow vertical line in the middle represents the border between the first 20 and the last 20 predicted frames by various models.

We can see B-F achieves better performance than B-B in short term prediction task when actual input frames are provided, but B-B outperforms B-F in the longer range prediction, reflecting learning of the relationships at the movement levels by the 3D convLSTM.

B-F doing better than F-F confirmed that the spatiotemporal block data structure provides additional information for modeling movement tendency.

Finally, we found that even F-F achieved better prediction results than PredNet, suggesting that a feature hierarchy might be more useful than a hierarchy of predicted errors.

Finally, our B-B network outperformed the state-of-the-art PredRNN++.

BID40 introduced the KTH video database which contains 2391 sequences of six human actions: walking, jogging, running, boxing, hand waving, and hand clapping, performed by 25 subjects in four different scenarios.

We divided video clips across all 6 action categories into a training set of 108717 sequences (persons #1-16) and a test set of 4086 sequences (persons #17- -block (B-B) , block-toframe (B-F), frame-to-frame (F-F)), PredNet, and PredRNN++, respectively.

k=1 to k=19 are predicted frames of the models when the input frames were available.

k=21 to k=39 are the "deadreckoning" predicted frames of the model when there are no input.

25) as was done in , except we extracted 40-frame sequences.

We center-cropped each frame to a 120×120 square and then re-sized it to input frame size of 64×64.

TAB1 compared the results of the different models on the KTH dataset, essentially reproducing all the observations we made based on the Moving-MINST dataset FIG2 .

B-B outperformed all tested models in the long range video prediction task.

FIG5 (a) and (b) compared the video prediction performance of the different models in terms of the "dead-reckoning frames" to be predicted when only the first twenty frames were provided for the two datasets.

The results show that, in both cases, B-B is far more effective than B-F in long range video prediction.

Hierarchical feedback in HPNet endows the representations in the earliest Cortical Modules with sensitivity to global movement and image patterns, despite these units' very localized receptive fields, particularly R in the feedforward path ( FIG7 ).

Could the neurons in the early visual areas of the mammalian hierarchical visual systems behave in a similar way, becoming sensitive to the memory of global movement patterns of familiar movies?We found this to be the case in a series of neurophysiological experiments that we have performed to study the effect of unsupervised learning of video sequences on the early visual cortical representations.

Two monkeys, implanted with Gray-Matter semi-chronic multielectrode arrays (SC32 and SC96) over the V1 operculum with access to neurons in V1 and V2, participated in the experiment.

Each experiment lasted for at least seven daily recording sessions.

In each recording session, the monkey was required to fixate on a red dot on the screen for a water reward while a set of 40 video clips of natural scenes with global movement patterns was presented.

One clip was presented per trial.

Each clip lasted for 800 ms.

A total of 40 clips were presented once each in a random interleaved fashion in a block of trials, and each block was repeated 20-25 times each day 3 .

Among the 40 movie clips tested every day, twenty of these were the same each day, designated as "Predicted set".

Twenty of them were different each day, designated as "Unpredicted set".

Each set consisted of 20 movies.

The rationale for the experimental design is as follows.

Given that we were recording from 30+ neurons in each session, even though the neurons have different stimulus preferences in their local receptive fields, each neuron would experience about 400 movie frames for the Predicted movie set, as well as for each of the Unpredicted movie sets.

When we averaged the temporal responses of all the neurons to each of the 20-movie sets, they should be roughly the same.

In the first two days of the experiment, the clips in the Predicted set were still unpredicted, hence there should have been no difference between the population averaged responses to the Predicted set and the Unpredicted set.

This was indeed the case as shown in FIG8 (top row) which compared the averaged temporal responses of the neurons to the Predicted set and to the Unpredicted set for the first two days of training in one experiment.

DISPLAYFORM0 Interestingly, we found that after only three days of unsupervised training, with 20-25 exposures of each familiar movie per day, the neurons started to respond significantly less to predicted movies than to novel movies in the later part of their responses, starting around 100 ms post-stimulus onset, as shown in FIG8 (b) (bottom row).

The evolution of daily mean of all neurons' familiarity suppression index over days is shown as the magenta curve.

As the neurons became more and more familiar with the Predicted set, the prediction suppression effect gradually increased and saturated at around the sixth and seventh days.

We repeated the experiments six times in two monkeys and obtained fairly consistent results.

Note that the movie clips were shown in a 8 o aperture during the experiment.

Given that the V1 and V2 neurons being studied have very local and small receptive fields (0.5 o to 2 o ), it is rather improbable that the neurons would have remembered or adapted to the local movement patterns of the Predicted set within their receptive fields, as they would be experiencing millions of such local spatiotemporal patterns in their daily experience.

Indeed, when the video clips were shown to the neurons through a smaller 3 o diameter aperture, the prediction suppression effects were much attenuated, suggesting that the neurons had indeed became sensitive to the global context of movement patterns!

To check whether neurons in our network behave in the same way, we performed a similar experiment on our network, pretrained with the KTH dataset.

We randomly extracted 20 sequences from the BAIR dataset BID8 , resized the sequence length to 40 frames and each frame size to 64×64.

We separated the 20 video sequences into two sets -the Predicted set and the Unpredicted set.

We averaged the responses to the two movie sets respectively of each type of neurons in the network (E (prediction error units), P (prediction units), and R (representation units)) in each CM within the center 8×8 hypercolumns.

Before training, the responses of each type of neurons are indeed the same for both movie sets (not shown, but similar to FIG8 .

Then, we trained the network with the Predicted set for 2000 epochs.

After training, all three types of units in each CM exhibited the prediction suppression effect as shown in FIG8 (c)-(h) (full details in Appendix C).

We observed the prediction suppression effect in all three types of neurons in all the modules in the hierarchy, with the higher modules showing a stronger effect.

It is not surprising that the prediction error neurons E would decrease their responses as the network learns to predict the familiar movies better.

It is rather interesting to find the representation neurons R and the prediction neurons P also exhibit prediction suppression, even though these neurons represent features rather than prediction errors.

The precise reasons remain to be determined, but the fact that all neuron types in the model exhibited the prediction suppression effect might explain why the prediction suppression effects were commonly observed in most of the randomly sampled neurons in the visual cortex (see FIG8 ).

These findings suggest that (1) predictive self-supervised learning might indeed be an important principle and mechanism by which the visual cortex learns its representations, and (2) the neurophysiological observations on prediction suppression in IT (see Appendix D) and now in the early visual cortex might be explained by this class of hierarchical cortical models.

In this paper, we developed a hierarchical prediction network model (HPNet), with a fast DCNN feedforward path, a feedback path and local recurrent LSTM circuits for modeling the counterstream / analysis-by-synthesis architecture of the mammalian hierarchical visual systems.

HPNet utilizes predictive self-supervised learning as in PredNet and PredRNN++, but integrates additional neural constraints or theoretical neuroscience ideas on spatiotemporal processing, counter-stream architecture, feature hierarchy, prediction evaluation and sparse convolution into a new model that delivers the state-of-the-art performance in long range video prediction.

Most importantly, we found that the hierarchical interaction in HPNet introduces sensitivity to global movement patterns in the representational units of the earliest module in the network and that real cortical neurons in the early visual cortex of awake monkeys exhibit very similar sensitivity to memories of global movement patterns, despite their very local receptive fields.

These findings support predictive self-supervised learning as an important principle for representation learning in the visual cortex and suggest that HPNet might be a viable computational model for understanding the cortical circuits in the hierarchical visual system at the functional level.

Further evaluations are needed to determine definitively whether PredNet or HPNet is a better fit to the biological reality.

APPENDIX A 3D CONVOLUTIONAL LSTM Because our data are in the unit of spatitemporal block, we have to use a 3D form of the 2D convolutional LSTM.

3D convolutional LSTM has been used by BID1 in the stereo setting.

The dimensions of the input video or the various representations (I, E and H) in any module are c×d×h×w, where c is the number of channels, d is the number of adjacent frames, h and w specify the spatial dimensions of the frame.

The 3D spatiotemporal convolution kernel is m × k × k in size, where m is kernel temporal depth and k is kernel spatial size.

The spatial stride of the convolution is 1.

The size of the output with n kernels is n × d × h × w.

We define the inputs as X 1 , ..., X t , the cell states as C 1 , ..., C t , the outputs as H 1 , ..., H t , and the gates as i t , f t , o t .

Our 3D convolutional LSTM is specified by the equations below, where the function of 3D convolution is indicated by and the Hadamard product is indicated by •. FIG7 of the main text of the paper.

The figures demonstrate that as more higher order modules are stacked up in the hierarchy, the semantic clustering into the six movement classes become more pronounced even in the early modules, suggesting that the hierarchical interaction has steered the feature representation into semantic clusters even in the early modules.

Module 4-1 means representation of module 1 in a 4-module network.

DISPLAYFORM0 We use linear decoding (multi-class SVM) to assess the distinctiveness of the semantiuc clusters in the representation of the different modules in the different networks.

The decoding results in TAB2 shows that the decoding accuracy based on the reprsentation of module 1 has improved from chance (16%) to 26%, an improvement of 60% between a 1-module HPNet and a 4-module HPNet, and that the representation of module 4 of a 4-module HPNet can achieve a 63% accuracy in classifying the six movement classes, suggesting that the network only needs to learn to predict unlabelled video sequences, and it automatically learns reasonable semantic representations for recognition.

For comparison, we also performed decoding on the output representations of each LSTM layer in the PredRNN++ and PredNet to study their representations of the six movement patterns.

The results shown below indicate that the semantic clustering of the six movements is not very strong in the PredRNN++ hierarchy.

We realized that this might be because the PredRNN++ behaves essentially like an autoencoder.

The four-layer network effectively only has two layers of feature abstraction, with layer 2 being the most semantic in the hierarchy and layers 3 and 4 representing the unfolding of the feedback path.

Decoding results indicate that the hierarchical representation based on the output of the LSTM at every layer in PredNet, which serve to predict errors of prediction errors of the previous layer, does not contain semantic information about the global movement patterns.

Figure 8 : Results of video sequence learning experiments showing prediction suppression can be observed in E, P , and R units in every module along the hierarchical network.

The abscissa is time after stimulus onset -where we set each video frame to be 25 ms for comparison with neural data.

The ordinate is the normalized averaged temporal response of all the units within the center 8×8 hypercolumns, averaged across all neurons and across the 20 movies in the Predicted set (blue) and the Unpredicted set (red) respectively.

Prediction suppression can be observed in all types of units, though more pronounced in the E and P units.

HPNet readily reproduces the prediction suppression effects observed in IT neurons.

BID29 trained monkeys to image pairs in a fixed order for over 800 trials for each 8 pair images, and then compared the responses of the neurons to these images in the trained order against the responses of the neurons to the same images but in novel pairings.

FIG11 shows the mean responses of 81 IT neurons during testing stage for predicted pairs and unpredicted pairs.

All the stimuli are presented in both pairs.

They found that neural responses to the expected second images in a familiar sequence order is much weaker than the neural responses to the image in an unfamiliar or unexpected sequence order.

To evaluate whether HPNet can produce the same effect, we performed exactly the same experiments with 2000 epochs of training on the image pairs, with a gap of 2 frames, and our model produced the same results, with lower responses for the predicted second stimulus relative to the unpredicted second stimulus.

Each stimulus sequence was presented first with 5 gray frames, followed by 10 frames of the first image in the pair, then 2 gray frames as gap, then 10 frames of the second image in the pair.

The responses of the units to the trained set and the untrained set are the same prior to training.

After training, the images when arranged in the trained order responded much less after the initial responses than the same images but arranged in unpredicted pairs.

The result shown in FIG1 duplicated the observations in BID29 , the average neural response of E unit is lower than the unpredicted pairs.

All three types of units of NPNet exhibit prediction suppression though the effect is much weaker for the R units (see FIG1 .

BID25 also tested the prediction suppression effect, but their model couldn't allow any gap between the stimuli as in the experiment.

Our model can handle gap because of our model is processing information in spatiotemporal blocks.

@highlight

A new hierarchical cortical model for encoding spatiotemporal memory and video prediction