We present a real-time method for synthesizing highly complex human motions using a novel training regime we call the auto-conditioned Recurrent Neural Network (acRNN).

Recently, researchers have attempted to synthesize new motion by using autoregressive techniques, but existing methods tend to freeze or diverge after a couple of seconds due to an accumulation of errors that are fed back into the network.

Furthermore, such methods have only been shown to be reliable for relatively simple human motions, such as walking or running.

In contrast, our approach can synthesize arbitrary motions with highly complex styles, including dances or martial arts in addition to locomotion.

The acRNN is able to accomplish this by explicitly accommodating for autoregressive noise accumulation during training.

Our work is the first to our knowledge that demonstrates the ability to generate over 18,000 continuous frames (300 seconds) of new complex human motion w.r.t.

different styles.

The synthesis of realistic human motion has recently seen increased interest BID12 BID39 BID4 BID14 BID1 BID25 with applications beyond animation and video games.

The simulation of human looking virtual agents is likely to become mainstream with the dramatic advancement of Artificial Intelligence and the democratization of Virtual Reality.

A challenge for human motion synthesis is to automatically generate new variations of motions while preserving a certain style, e.g., generating large numbers of different Bollywood dances for hundreds of characters in an animated scene of an Indian party.

Aided by the availability of large human-motion capture databases, many database-driven frameworks have been employed to this end, including motion graphs BID18 BID33 BID27 , as well as linear BID34 BID2 BID36 and kernel methods BID29 BID31 BID8 BID28 BID42 , which blend key-frame motions from a database.

It is hard for these methods, however, to add new variations to existing motions in the database while keeping the style consistent.

This is especially true for motions with a complex style such as dancing and martial arts.

More recently, with the rapid development in deep learning, people have started to use neural networks to accomplish this task BID13 .

These works have shown promising results, demonstrating the ability of using high-level parameters (such as a walking-path) to synthesize locomotion tasks such as jumping, running, walking, balancing, etc.

These networks do not generate new variations of complex motion, however, being instead limited to specific use cases.

In contrast, our paper provides a robust framework that can synthesize highly complex human motion variations of arbitrary styles, such as dancing and martial arts, without querying a database.

We achieve this by using a novel deep auto-conditioned RNN (acRNN) network architecture.

Recurrent neural networks are autoregressive deep learning frameworks which seek to predict sequences of data similar to a training distribution.

Such a framework is intuitive to apply to human motion, which can be naturally modeled as a time series of skeletal joint positions.

We are not the first to leverage RNNs for this task BID4 BID14 BID1 BID25 , and these works produce reasonably realistic output at a number of tasks such as sitting, talking, smoking, etc.

However, these existing methods also have a critical drawback: the motion becomes unrealistic within a couple of seconds and is unable to recover.

This issue is commonly attributed to error accumulation due to feeding network output back into itself BID13 .

This is reasonable, as the network during training is given ground-truth input sequences to condition its subsequent guess, but at run time, must condition this guess on its own output.

As the output distribution of the network will not be identical to that of the ground-truth, it is in effect encountering a new situation at test-time.

The acRNN structure compensates for this by linking the network's own predicted output into its future input streams during training, a similar approach to the technique proposed in BID0 .

Our method is light-weight and can be used in conjunction with any other RNN based learning scheme.

Though straightforward, this technique fixes the issue of error accumulation, and allows the network to output incredibly long sequences without failure, on the order of hundreds of seconds (see Figure 5 ).

Though we are yet as unable to prove the permanent stability of this structure, it seems empirically that motion can be generated without end.

In summary, we present a new RNN training method capable for the first time of synthesizing potentially indefinitely long sequences of realistic and complex human motions with respect to different styles.

Many approaches have developed over the years in order to generate realistic human motion.

In this section, we first review the literature that has dealt with motion synthesis using simulation methods and database-driven methods, then review the more recent deep learning approaches.

Simulation-based Methods.

Simulation-based techniques are able to produce physically plausible animations BID5 BID3 BID10 BID9 BID24 , including realistic balancing, motion on terrain of various heights, and recovery from falling.

Many of these methods consider physical constraints on the human skeleton while optimizing an motion objective.

For example, in the work of , one task they employ is moving the skeleton in a certain direction without falling over.

Similarly, in Ha et al. BID9 , given initial fall conditions, they seek to minimize joint stress due to landing impact while ensuring a desired landing pose.

Though the output is plausible, they require specific objectives and constraints for each individual task; it is infeasible to apply such explicit priors to highly stylized and varied motion such as dancing or martial arts.

There are also some recent works BID32 BID26 which attempt to use less rigid objectives, instead employing adversarial and reinforcement learning.

However, these motions often look uncanny and not human-like.

Database-driven Methods.

Motion graphs BID18 BID33 BID27 , which stitch transitions into segments queried from a database, can generate locomotion along arbitrary paths, but are in essence limited to producing memorized sequences.

Frameworks based on linear BID34 BID2 BID36 and kernal methods BID29 BID31 BID8 BID28 BID42 have also shown reasonable success.

Taylor et al. BID37 use conditional restricted boltzmann machines to model motion styles.

Grochow et al. BID8 and BID41 both use Gaussian Process Latent Variable Models (GPLVM) to control motion.

apply reinforcement learning in a reduced space to compute optimal motions for a variety of motion tasks, including movement, punching, and kicking.

Kee et al. BID20 use a representation of motion data called motion-fields which allows users to interactively control movement of a character.

The work of Xia et al. BID44 is able to achieve real-time style transfers of unlabeled motion.

Taylor et al. BID38 use binary latent variables which are connected between different time steps while Liu et al. BID23 estimate physical parameters from motion capture data.

Database methods have limitations in synthesizing new variations of motion, in addition to the high memory costs for storing complex motions.

Deep Learning Approaches.

The use of recurrent networks is a natural approach to dealing with the problem of human motion.

RNNs are trained to generate output sequentially, each output conditioned on the previous elements in the sequence.

These networks have shown much success in Natural Language Processing for generating text BID35 , hand written characters BID6 BID7 , and even captioning images BID40 .

For the purpose of motion prediction and generation, Fragkiadaki et al. BID4 propose to use encoderrecurrent-decoder (ERD), jointly learning a skeleton embedding along with sequential information, in order to make body positions more realistic.

In Jain et al. BID14 , the authors employ RNNs to learn spatio-temporal graphs of interaction, a structure which naturally applies to the human skeleton over time-joint positions are related over consecutive frames, and joints also interact with each other spatially (arms and legs interact with each other, the spine interacts with all other joints).

B??tepage et al. BID1 attempt to learn dance sequentially, but they are unable to produce varied and realistic output.

More recently, Martinez et al. BID25 propose using a sequence-to-sequence architecture along with sampling-based loss.

A main problem with these approaches, even in the case of the ERD, is that motion generation tends to converge to a mean position over time.

Using our proposed training method, motion neither halts nor becomes unrecognizable for any period of time.

Holden et al. BID11 demonstrate that a manifold of human motion can be learned using an autoencoder trained on the CMU dataset.

They BID12 extend this work by using this deep convolutional auto-encoder, in addition to a disambiguation network, in order to produce realistic walking motion along a user-defined path.

They are also able to use the embedding space of the encoder to perform style transfer across different walks.

Using the same autoencoder and a different disambiguation network, they can also perform other user defined tasks such as punching and kicking.

Their method does not use an RNN structure and so it can only generate fixed length sequences based on the architecture of the disambiguation network.

More recently, Holden et al. BID13 ) use a Phase-Functioned Neural Network which takes the geometry of the scene into account to produce motion along a user-defined path.

This method makes use of humans' periodic change of gait to synthesize realistic walking motion but has not demonstrated the ability to generate motions that have more complex step movements.

Auto-Conditioned RNN.

Recurrent neural networks are well documented in the literature.

A good survey and introduction can be found at BID16 BID30 .

In short, a recurrent network can be represented as a function with a hidden memory unit, x t+1 = f (x t , m t ), where m t is the "memory" of the network, and is updated during every forward pass, and initialized at 0.

The motivation is that the memory stores important information about a sequence up until that point, which can help with the prediction of the next element.

In the experiments that follow, we use a special type of RNN called an "LSTM", or a "long short term memory" network.

We refer to the network trained with our method as "acLSTM".

As mentioned in the introduction, the major drawback of using LSTM/RNN deep learning methods for motion prediction is the problem of error accumulation.

Following the conventional way to train an RNN, the network is recursively given a sequence of ground truth motion data, DISPLAYFORM0 and asked to produce the output G 2,k+1 [g 2 , ..., g k+1 ].

Specifically, at training time, the recursive module at time step t, M t , is conditioned on the input [g 1 , ..., g t???1 ], and the error is measured between its output and g t .

Because of this, the backpropogation algorithm BID43 used for updating the parameters will always optimize w.r.t.

the input ground-truth sequences [g 1 , ..., g t???1 ].

The parameters are accustomed to ground truth input -something it does not have access to at test-time.

It is easy to see why problems will emerge: even if initial input is similar to ground truth, those slight differences will accumulate as the output is fed back in, producing output that become progressively worse until the sequence diverges or freezes.

Effectively, the network is encountering a completely novel situation during test time as compared to training time, and so cannot perform well.

The issue is so prevalent, in fact, that previous methods fail to produce realistic output after just several seconds BID4 BID14 BID25 .Holden et al. BID11 show that an autoencoder framework can to some degree "fix" such broken input, and some researchers have tried jointly learning such an encoder-decoder network alongside RNNs to better condition subsequent input BID7 BID4 .

However, in the case of this framework being applied to motion as in ERD BID4 , it does not generalize to indefinitely long sequences, as shown in BID14 .

It seems as though the autoencoder might mitigate error accumulation, but does not eliminate it.

The acRNN, on the other hand, deals with poor network output explicitly by using it during training.

Instead of only feeding in ground-truth instances, we use subsequences of the network's own outputs at periodic intervals.

For instance, sticking with the example above, instead of conditioning the network on DISPLAYFORM1 The variable p u+1 is the network output conditioned on [g 1 , ..., g u ], and p u+2 is conditioned on [g 1 , ..., g u , p u+1 ].

In this example, we refer to v as the "condition length" and u as the "ground-truth length".

As the network is conditioned on its own output during training, it is able to deal with such input during synthesis.

FIG0 details an unrolled Auto-Conditioned RNN with condition length u = v = 4, and FIG0 shows a more detailed view or our network.

The method of BID0 also proposes using network output during training, but does so stochastically, without fixing condition lengths.

However, we found that changing the condition/ground-truth length while keeping the proportion of ground-truth input fixed affects both the accuracy and variation of the output.

See Figure 9 in the appendix.

Auto-conditioning also has the interpretation of training the network to produce longer sequences without further input.

Whereas with standard training the network error is measured only against p u+1 when conditioned with [g 1 , ..., g u ], under auto-conditioning the error is computed on the entire sequence p u+1 , ..., p u+v w.r.t.

the same input.

This effectively forces the network to produce v frames of output simultaneously as opposed to only one.

Martinez et al. BID25 ) also use contiguous sequences of network output during training, but unlike us they do not alternate these with ground-truth input at regular intervals.

Data Representation.

We use the publicly available CMU motion-capture dataset for our experiments.

The dataset is given as sequences of 57 skeleton joint positions in 3D-space.

First, we define a root joint, whose position at time t is given by r t = (r 1,t , r 2,t , r 3,t ).

In order to better capture relative motion, we instead use the displacement of the root from the previous frame for input -r t = (r 1,t ??? r 1,t???1 , r 2,t ??? r 2,t???1 , r 3,t ??? r 3,t???1 ).

For every other joint at time t, with position j t = (j 1,t , j 2,t , j 3,t ), we represent it as as the relative distance in the world-coordinate system to the root joint,j t = (j 1,t ??? r 1,t , j 2,t ??? r 2,t , j 3,t ??? r 3,t ).

All distances are stored in meters.

We use a skeleton with height 1.54 meters in neutral pose for all experiments.

We found this representation to be desirable for several reasons.

Primarily, if there is periodic motion in the dataset, we would like frames at the same point in the repeated activity to have small Euclidean distance.

If we instead used absolute positions, even if it were only for the hip, this would certainly not be the case.

We note that there are alternative representations which achieve the same property.

BID4 express joint positions as rotations relative to a parent joint, and BID12 define them in the body's relative coordinate system along with a relative rotation of the body w.r.t.

the previous frame.

Training.

We train the acLSTM with three fully connected layers with a memory size of 1024, similar to BID4 BID14 BID1 .

The main difference is that for every u ground-truth inputs of the time series, we connect v instances of the network's own output into its subsequent input streams (see section 2.1).

In the main body of the paper, we set u = v = 5.

We carry out further experiments with varying u and v in the appendix.

We train with a sequence length of 100 for 500000 iterations using the ADAM backpropogation algorithm BID17 on an NVIDIA 1080 GPU for each dataset we experiment on.

We use Euclidean loss for our objective function.

The initial learning rate is is set to 0.0001.

We implement the training using the python caffe framework BID15 .

We sample sequences at multiple frame-rates as well as rotate the sequence randomly in order to increase the training size.

In detail, if at time t we input the network H with ground truth, then the loss is given by: DISPLAYFORM2 (1) where x t and x t+1 are the ground truth motions for time steps t and t + 1.If at time t the network is input with its own previous output, the loss is given by: DISPLAYFORM3 DISPLAYFORM4 .

k indicates how many times the network has fed itself is its own input since the last injection of ground-truth.

It is bounded by the condition length (see previous section).

We evaluate our synthesized motion results on different networks trained on each of four distinct subsets from the CMU motion capture database: martial arts, Indian dance, Indian/salsa hybrid, and walking.

An anonymous video of the results can be found here: https://youtu.be/FunMxjmDIQM.Quantitative Results.

TAB0 shows the prediction error as Euclidean distance from the ground truth for different motion styles at various time frames.

We compare with a 3-layer LSTM (LSTM-3LR), ERD BID4 , the seq2seq framework of BID25 as well as scheduled sampling BID0 .

Though our goal is long-term stable synthesis and not prediction, it can be seen in TAB0 that acLSTM performs reasonably well in this regard, even achieving the best performance at most time-scales for the Indian dance category.

The method of BID25 performs the best in the short term, but has the worst error by far past the half-second mark.

As noted in BID4 , the stochasticity of human motion makes longer-term prediction infeasible.

TAB3 in the appendix shows this error difference between versions of the Indian dance network trained with different condition lengths.

Figure 2 shows the average change between subsequent frames for different frame times for the acLSTM and the basic scheme.

For the more complex motions of Martial Arts and Indian Dance, it is clear from the graph that acLSTM continues producing motion in the long-term while the basic training scheme (LSTM-3LR) without auto-conditioning results in stagnated motion, as the network freezes into a converged mean position.

Likewise, Figure 9 in the appendix shows this average change for the Indian dance network trained with different condition lengths.

We note that while the methods of BID4 BID25 do not simply freeze completely, as with the basic scheme, their motion becomes unrealistic at around the same time FIG3 .

This is consistent with the observations of the original authors.

Qualitative Results.

Figure 4 shows several example frames taken from 50 second synthesized outputs, representing both the extended long term complexity and plausibility of the output.

In comparison, our implementations of ERD, and seq2seq are only able to generate motion for a couple :DONLQJ /670/5 DF/670/5ZFOHQJWK ,QGLDQ'DQFH )UDPH1XPEHU 0DUWLDO$UWV of seconds before they become unrealistic FIG3 ).

Scheduled sampling BID0 performs better than ERD and seq2seq, but also freezes eventually, perhaps because it does not force the network to generate consistently longer sequences during training as our method does.

We also demonstrate the possibility of creating hybrid motions by mixing training sets in third row of Figure 4 .It should be noted that the motion in our framework, while never permanently failing, also does not remain perfectly realistic, and short-term freezing does sometimes occur.

This, however, does not occur for the network trained just on walking.

It is perhaps that the movement of the feet in conjunction with the absolute movement of the root joint is not as easy to discern when the feet leave the ground aperiodically, or there are jumping motions.

BID4 0 BID4 0.32 0.44 0.63 0.90 1.14 1.40 1.61 1.88 seq2seq BID25 0 Figure 4: Motion sequences generated by acLSTM, sampled at various frames.

Motion style from top to bottom: martial arts, Indian dancing, Indian/salsa hybrid and walking.

All the motions are generated at 60 fps, and are initialized with 10 frames of ground truth data randomly picked up from the database.

The number at the bottom of each image is the frame index.

The images are rendered with BVHViewer 1.1 BID39 When motion does stagnate, it recovers relatively quickly, and the motion never diverges or freezes completely (see Figure 6 ).

The short-term freezing could possibly be explained by "dead-times" in the training sequence, where the actor is beginning or ending a sequence which involves a rest position.

Note that when training with two different datasets, as in the case of the Indian/Salsa combined network, motion borrows from both dance styles.

We also demonstrate in Figure 5 that our method does not freeze even after 20,000 frames of synthesis, which is approximately 333 seconds of output.

One can see a qualitative comparison of acLSTM with a basic LSTM-3LR in Figure 3 , both trained on the Indian dance dataset.

We find the performance of the vanilla network to be consistent with the results reported in BID4 BID14 BID1 BID25 , freezing at around 1000 ms.

It never recovers from the motion.

Our network, on the other hand, continues producing varied motion for the same time frame.

We have shown the effectiveness of the acLSTM architecture to produce extended sequences of complex human motion.

We believe our work demonstrates qualitative state-of-the-art results in motion generation, as all previous work has focused on synthesizing relatively simple human motion for extremely short time periods.

These works demonstrate motion generation up to a couple of seconds at most while acLSTM does not fail even after over 300 seconds.

Though we are as of yet unable to prove indefinite stability, it seems empirically that acLSTM can generate arbitrarily long sequences.

Current problems that exist include choppy motion at times, self-collision of the skeleton, and unrealistic sliding of the feet.

Further developement of GAN methods, such as BID19 , could result in increased realism, though these models are notoriously hard to train as they often result in mode collapse.

Combining our technique with physically based simulation to ensure realism after synthesis is also a potential next step.

Finally, it is important to study the effects of using various condition lengths during training.

We begin the exploration of this topic in the appendix, but further analysis is needed.

Figure 9 might imply some sort of trade off between motion change over time and short-term motion prediction error when training with different condition lengths.

However, it is also possible that limiting motion magnitude on this particular dataset might correspond to lower error.

Further experiments of various condition lengths on several motion styles need to be conducted to say anything meaningful about the effect.

C VISUAL DIAGRAM OF AUTO-CONDITIONED LSTM

@highlight

Synthesize complex and extended human motions using an auto-conditioned LSTM network