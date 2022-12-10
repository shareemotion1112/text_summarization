Spiking neural networks are being investigated both as biologically plausible models of neural computation and also as a potentially more efficient type of neural network.

While convolutional spiking neural networks have been demonstrated to achieve near state-of-the-art performance, only one solution has been proposed to convert gated recurrent neural networks, so far.

Recurrent neural networks in the form of networks of gating memory cells have been central in state-of-the-art solutions in problem domains that involve sequence recognition or generation.

Here, we design an analog gated LSTM cell where its neurons can be substituted for efficient stochastic spiking neurons.

These adaptive spiking neurons implement an adaptive form of sigma-delta coding to convert internally computed analog activation values to spike-trains.

For such neurons, we approximate the effective activation function, which resembles a sigmoid.

We show how analog neurons with such activation functions can be used to create an analog LSTM cell; networks of these cells can then be trained with standard backpropagation.

We train these LSTM networks on a noisy and noiseless version of the original sequence prediction task from Hochreiter & Schmidhuber (1997), and also on a noisy and noiseless version of a classical working memory reinforcement learning task, the T-Maze.

Substituting the analog neurons for corresponding adaptive spiking neurons, we then show that almost all resulting spiking neural network equivalents correctly compute the original tasks.

With the manifold success of biologically inspired deep neural networks, networks of spiking neurons are being investigated as potential models for computational and energy efficiency.

Spiking neural networks mimic the pulse-based communication in biological neurons, where in brains, neurons spike only sparingly -on average 1-5 spikes per second BID0 .

A number of successful convolutional neural networks based on spiking neurons have been reported BID7 BID13 BID6 BID15 BID12 , with varying degrees of biological plausibility and efficiency.

Still, while spiking neural networks have thus been applied successfully to solve image-recognition tasks, many deep learning algorithms use recurrent neural networks (RNNs), in particular using Long Short-Term Memory (LSTM) layers BID11 .

Compared to convolutional neural networks, LSTMs use memory cells to store selected information and various gates to direct the flow of information in and out of the memory cells.

To date, the only spike-based version of LSTM has been realized for the IBM TrueNorth platform Shrestha et al.: this work proposes a method to approximate LSTM specifically for the constrains of this neurosynaptic platform by means of a store-and-release mechanism that synchronizes the modules.

This translates to a frame-based rate coding computation, which is less biological plausible and energy efficient than an asynchronous approach, as the one proposed here.

Here, we demonstrate a gated recurrent spiking neural network that corresponds to an LSTM unit with a memory cell and an input gate.

Analogous to recent work on spiking neural networks (O 'Connor et al., 2013; BID6 BID19 BID20 , we first train a network with modified LSTM units that computes with analog values, and show how this LSTMnetwork can be converted to a spiking neural network using adaptive stochastic spiking neurons that encode and decode information in spikes using a form of sigma-delta coding BID18 BID19 BID14 .

In particular, we develop a binary version of the adaptive sigma-delta coding proposed in BID19 : we approximate the shape of the transfer function that this model of fast-adapting spiking neurons exhibits, and we assemble the analog LSTM units using just this transfer function.

Since input-gating is essential for maintaining memorized information without interference from unrelated sensory inputs BID11 , and to reduce complexity, we model a limited LSTM neuron consisting of an input cell, input gating cell, a Constant Error Carousel (CEC) and output cell.

The resultant analog LSTM network is then trained on a number of classical sequential tasks, such as the noise-free and noisy Sequence Prediction and the T-Maze task BID11 BID1 .

We demonstrate how nearly all the corresponding spiking LSTM neural networks correctly compute the same function as the analog version.

Note that the conversion of gated RNNs to spike-based computation implies a conversion of the neural network from a time step based behavior to the continuous-time domain: for RNNs, this means having to consider the continuous signal integration in the memory cell.

We solve the time conversion problem by approximating analytically the spiking memory cell behavior through time.

Together, this work is a first step towards using spiking neural networks in such diverse and challenging tasks like speech recognition and working memory cognitive tasks.

To construct an Adapting Spiking LSTM network, we first describe the Adaptive Spiking Neurons and we approximate the corresponding activation function.

Subsequently, we show how an LSTM network comprised of a spiking memory cell and a spike-driven input-gate can be constructed and we discuss how analog versions of this LSTM network are trained and converted to spiking versions.

Adaptive Spiking Neuron.

The spiking neurons that are used in this paper are Adaptive Spiking Neurons (ASNs) as described in BID2 .

This is a variant of an adapting Leaky Integrate & Fire (LIF) neuron model that includes fast adaptation to the dynamic range of input signals.

The ASNs used here communicate with spikes of a fixed height h = 1 (binary output), as suggested by BID20 .

The behavior of the ASN is determined by the following equations: incoming postsynaptic current: DISPLAYFORM0 input signal: DISPLAYFORM1 threshold: DISPLAYFORM2 internal state: DISPLAYFORM3 where w i is the weight (synaptic strength) of the neuron's incoming connection; t i s < t denote the spike times of neuron i, and t s < t denote the spike times of the neuron itself; φ(t) is an exponential smoothing filter with a short time constant τ φ ; ϑ 0 is the resting threshold; m f is a variable controlling the speed of spike-rate adaptation; τ β , τ γ , τ η are the time constants that determine the rate of decay of I(t), ϑ(t) andŜ(t) respectively (see BID2 and BID19 for more details).

As in BID2 , the ASN emits spikes following a stochastic firing condition defined as: DISPLAYFORM4 where V (t) is the membrane potential defined as the difference between S(t) andŜ(t), λ 0 = 0.005 is a normalization parameter and ∆V = 0.1 is a scaling factor that defines the slope of the stochastic area.

Activation function of the Adaptive Analog Neuron.

In order to create a network of ASNs that performs correctly on typical LSTM tasks, our approach is to train a network of Adaptive Analog Neurons (AANs) and then convert the resulting analog network into a spiking one, similar to O 'Connor et al. (2013); BID6 ; BID19 .

We define the activation function of the AANs as the function that maps the input signal S to the average PSC I that is perceived by the next (receiving) ASN in a defined time window.

We normalize the obtained spiking activation function at the point where it reaches a plateau.

We then fit the normalized spiking activation function with a sum-of-exponentials shaped function as: DISPLAYFORM5 with derivative: DISPLAYFORM6 where, for the neuron parameters used, we find a = 148.7, b = −10.16, c = 3.256 and d = −1.08.Using this mapping from the AAN to the ASN (see Figure 1 ), the activation function can be used during training: thereafter, the ASNs are used as "drop in" replacements for the AANs in the trained network.

Unless otherwise stated, the ASNs use τ η = τ β = τ γ = 10 ms, and ϑ 0 and m f are set to 0.3 and 0.18 for all neurons.

The spike height, h, is found such that ASN(4.8) = 1.

Note that the spike height h is a normalization parameter for the activation function of the ASN model: in order to have binary communication across the network, the output weights are simply scaled by h.

Adaptive Spiking LSTM.

An LSTM cell usually consists of an input and output gate, an input and output cell and a CEC BID11 .

Deviating from the original formulation, and more recent versions where forget gates and peepholes were added Gers et al. FORMULA1 , the Adaptive Spiking LSTM as we present it here only consists of an input gate, input and output cells, and a CEC.

As noted, to obtain a working Adaptive Spiking LSTM, we first train its analog equivalent, the Adaptive Analog LSTM.

Figure 2 shows the schematic of the Adaptive Analog LSTM and its spiking analogue.

It is important to note that we aim for a one-on-one mapping from the Adaptive Analog LSTM to the Adaptive Spiking LSTM.

This means that while we train the Adaptive Analog LSTM network with the standard time step representation, the conversion to the continuous-time spiking domain is achieved by presenting each input for a time window of size ∆t.

Sigmoidal ASN.

The original formulation of LSTM uses sigmoidal activation functions in the input gate and input cell.

However, the typical activation function of real neurons resembles a half-sigmoid and we find that the absence of a gradient for negative input is problematic during training.

Here, we approximate a sigmoidal-shaped activation function by exploiting the stochastic firing condition of the ASN.

Indeed, Figure 1 shows that the ASN has a non-null probability to fire even under the threshold ϑ 0 .

Therefore, the AAN transfer function of Eq. 6 holds a gradient in that area.

Together with the maximal activation being normalized to 1 (see Eq. 6 for lim S→∞ ) the AAN transfer function represents a good candidate for LSTM operations such as closing and opening the gates.

Spiking input gate and spiking input cell.

The AAN functions are used in the Adaptive Analog LSTM cell for the input gate and input cell.

The activation value of the input cell is multiplied by the activation value of the input gate, before it enters the CEC, see Figure 2 .

In the spiking version of the input gate, the outgoing signal from the ASN is accumulated in an intermediate neuron (ASN * in Figure 2 ).

The internal stateŜ of this neuron is then multiplied with the spikes that move from the ASN of the input cell to the ASN of the output cell.

This leads to a direct mapping from the Adaptive Analog LSTM to the Adaptive Spiking LSTM.Spiking Constant Error Carousel (CEC) and spiking output cell.

The Constant Error Carousel (CEC) is the central part of the LSTM cell and avoids the vanishing gradient problem BID11 .

In the Adaptive Spiking LSTM, we merge the CEC and the output cell to one ASN with an internal state that does not decay -in the brain could be implemented by slowly decaying (seconds) neurons BID5 .

The value of the CEC in the Adaptive Analog LSTM corresponds with state I of the ASN output cell in the Adaptive Spiking LSTM.In the Adaptive Spiking LSTM, we set τ β in Equation 1 to a very large value for the CEC cell to obtain the integrating behavior of a CEC.

Since no forget gate is implemented this results in a spiking CEC neuron that fully integrates its input.

When τ β is set to ∞, every incoming spike is added to a non-decaying PSC I.

So if the state of the sending neuron (ASN in in FIG1 ) has a stable inter-spike interval (ISI), then I of the receiving neuron (ASN out ) is increased with incoming spike height h every ISI, so h ISI per time step.

For a stochastic neuron, this corresponds to the average increase per time step.

The same integrating behavior needs to be translated to the analog CEC.

Since the CEC cell of the Adaptive Spiking LSTM integrates its input S every time step by S τη , we can map this to the CEC of the Adaptive Analog LSTM.

The CEC of a traditional LSTM without a forget gate is updated every time step by CEC(t) = CEC(t − 1) + S, with S its input value.

The CEC of the Adaptive Analog LSTM is updated every time step by CEC(t) = CEC(t − 1) + S τη .

This is depicted in Figure 2 via a weight after the input gate with value 1 τη .

To allow a correct continuous-time representation after the spike-coding conversion, we divide the incoming connection weight to the CEC, W CEC , by the time window ∆t.

In our approach then, we train the Adaptive Analog LSTM as for the traditional LSTM (without the τ η factor), which effectively corresponds to set a continuous-time time window ∆t = τ η .

Thus, to select a different ∆t, in the spiking version W CEC has to be set to W CEC = τ η /∆t.

The middle plot in FIG1 shows that setting τ β to ∞ for ASN out in a spiking network results in the same behavior as using an analog CEC that integrates with CEC(t) = CEC(t − 1) + S, since the slope of the analog CEC is indeed the same as the slope of the spiking CEC.

Here, every time step in the analog experiment corresponds to ∆t = 200 ms.

However, the spiking CEC still produces an error with respect to the analog CEC (the error increases for lower ∆ts, e.g. it doubles when going from 200ms to 50ms).

This is because of two reasons: first, the stochastic firing condition results in an irregular ISI; second, the adapting behavior of the ASN produces a transitory response that is not represented by the AAN transfer function.

For these reasons, by choosing bigger time windows ∆t more stable responses are obtained.

Learning rule used for training the spiking LSTM To train the analog LSTMs on the supervised tasks, a customized truncated version of real-time recurrent learning (RTRL) was used.

This is the same algorithm used in Gers et al. FORMULA1 , where the partial derivatives w.r.t.

the weights W xc and W xi (see Figure 2 ) are truncated.

For the reinforcement learning (RL) tasks we used RL-LSTM Bakker FORMULA1 , which uses the same customized, truncated version of RTRL that was used for the supervised tasks.

RL-LSTM also incorporates eligibility traces to improve training and Advantage Learning BID10 .

All regular neurons in the network are trained with traditional backpropagation.

Since the presented Adaptive Analog LSTM only has an input gate and no output or forget gate, we present four classical tasks from the LSTM literature that do not rely on these additional gates.

Sequence Prediction with Long Time Lags.

The main concept of LSTM, the ability of a CEC to maintain information over long stretches of time, was demonstrated in Hochreiter & Schmidhuber (1997) in a Sequence Prediction task: the network has to predict the next input of a sequence of p + 1 possible input symbols denoted as a 1 , ..., a p−1 , a p = x, a p+1 = y. In the noise free version of this task, every symbol is represented by the p + 1 input units with the i − th unit set to 1 and all the others to 0.

At every time step a new input of the sequence is presented.

As in the original formulation, we train the network with two possible sequences, (x, a 1 , a 2 , ..., a p−1 , x) and (y, a 1 , a 2 , ..., a p−1 , y), chosen with equal probability.

For both sequences the network has to store a representation of the first element in the memory cell for the entire length of the sequence (p).

We train 20 networks on this task for a total of 100k trials, with p = 100, on an architecture with p + 1 input units and p + 1 output units.

The input units are fully connected to the output units without a hidden layer.

The same sequential network construction method from the original paper was used to prevent the "abuse problem": the Adaptive Analog LSTM cell is only included in the network after the error stops decreasing BID11 .

In the noisy version of the sequence prediction task, the network still has to predict the next input of the sequence, but the symbols from a 1 to a p−1 are presented in random order and the same symbol can occur multiple times.

Therefore, only the final symbols a p and a p+1 can be correctly predicted.

This version of the sequence prediction task avoids the possibility that the network learns local regularities in the input stream.

We train 20 networks with the same architecture and parameters of the previous task, but for 200k trials.

For both noise-free and noisy tasks we considered the network converged when the average error over the last 100 trials was less than 0.25.T-Maze task.

In order to demonstrate the generality of our approach, we trained a network with Adaptive Analog LSTM cells on a Reinforcement Learning task, originally introduced in BID1 .

In the T-Maze task, an agent has to move inside a maze to reach a target position in order to BID11 and current implementation); while for the T-Maze tasks it corresponds to the total number of steps BID1 .

ASN accuracy (%), total number of spikes per task and firing rate (Hz) are also reported.

Note that the firing rate for both the sequence prediction tasks are computed without taking into account the input and output neurons not active in a specific time frame.

Task be rewarded while maintaining information during the trial.

The maze is composed of a long corridor with a T-junction at the end, where the agent has to make a choice based on information presented at the start of the task.

The agent receives a reward of 4 if it reaches the target position and −0.4 if it moves against the wall.

If it moves to the wrong direction at the T-junction it also receives a reward of −0.4 and the system is reset.

The larger negative reward value, w.r.t.

the one used in BID1 , is chosen to encourage Q-values to differentiate more during the trial.

The agent has 3 inputs and 4 outputs corresponding to the 4 possible directions it can move to.

At the beginning of the task the input can be either 011 or 110 (which indicates on which side of the T-junction the reward is placed).Here, we chose the corridor length N = 20.

A noiseless and a noisy version of the task were defined: in the noiseless version the corridor is represented as 101, and at the T-junction 010; in a noisy version the input in the corridor is represented as a0b where a and b are two uniformly distributed random variables in a range of [0, 1] .

While the noiseless version can be learned by LSTM-like networks without input gating BID16 , the noisy version requires the use of such gates.

The network consists of a fully connected hidden layer with 12 AAN units and 3 Adaptive Analog LSTMs.

To increase the influence of the LSTM cell in the network, we normalized the activation functions of the AAN output cell and ASN output cell at S = 1.

The same training parameters are used as in Bakker FORMULA1 ; we train 20 networks for each task and all networks have the same architecture.

As a convergence criteria we checked whenever the network reached on average a total reward greater than 3.5 in the last 100 trials.

As shown in TAB0 , all of the networks that were successfully trained for the noise-free and noisy Sequence Prediction tasks could be converted into spiking networks.

FIG3 shows the last 6 inputs of a noise-free Sequence Prediction task before (left) and after (right) the conversion, demonstrating the correct predictions made in both cases.

Indeed, for the 19 successful networks, after presenting either x or y as the first symbol of the sequence, the average error over the last 200ms was always below the chosen threshold of 0.25.

As it can be seen in Figure 6 , the analog and the spiking CEC follow a comparable trend during the task, reaching similar values at the end of the simulation.

Note that, in the noisy task, all the successfully trained networks were still working after the conversion: in this case, due to the input noise, the CEC values are always well separated.

Finally, we found that the number of trials needed to reach the convergence criterion were, on average, lower than the one reported in BID11 .Similar results were obtained for the T-Maze task: all the networks were successful after the conversion in both the noise-free and noisy conditions.

FIG4 shows the Q-values of a noisy T-Maze task, demonstrating the correspondence between the analog and spiking representation even in presence of noisy inputs.

However, we notice that the CEC of the spiking LSTMs reach different values compared to their analog counterparts.

This is probably due to the increased network and task complexity.

In general, we see that the spiking CEC value is close to the analog CEC value, while always exhibiting some deviations.

Moreover, TAB0 reports the average firing rate computed per task, showing reasonably low values compatible with the one recorder from real neurons.

Gating is a crucial ingredient in recurrent neural networks that are able to learn long-range dependencies BID11 .

Input gates in particular allow memory cells to maintain information over long stretches of time regardless of the presented -irrelevant -sensory input BID11 .

The ability to recognize and maintain information for later use is also that which makes gated RNNs like LSTM so successful in the great many sequence related problems, ranging from natural language processing to learning cognitive tasks BID1 .To transfer deep neural networks to networks of spiking neurons, a highly effective method has been to map the transfer function of spiking neurons to analog counterparts and then, once the network has been trained, substitute the analog neurons with spiking neurons O' Connor et al. (2013); BID6 ; BID19 .

Here, we showed how this approach can be extended to gated memory units, and we demonstrated this for an LSTM network comprised of an input gate and a CEC.

Hence, we effectively obtained a low-firing rate asynchronous LSTM network.

The most complex aspect of a gating mechanism turned out to be the requirement of a differentiable gating function, for which analog networks use sigmoidal units.

We approximated the activation function for a stochastic Adaptive Spiking Neurons, which, as many real neurons, approximates a half-sigmoid (Fig. 1) .

We showed how the stochastic spiking neuron has an effective activation even below the resting threshold ϑ 0 .

This provides a gradient for training even in that area.

The resultant LSTM network was then shown to be suitable for learning sequence prediction tasks, both in a noise-free and noisy setting, and a standard working memory reinforcement learning task.

The learned network could then successfully be mapped to its spiking neural network equivalent for at least 90% of the trained analog networks.

Figure 6 : The values of the analog CECs and spiking CECs for the noise-free Sequence Prediction (left, only one CEC cell was used) and noise-free T-maze (right, three CEC cells were used) tasks.

The spiking CEC is the internal stateŜ of the output cell of the Adaptive Spiking LSTM.We also showed that some difficulties arise in the conversion of analog to spiking LSTM.

Principally, the ASN activation function is derived for steady-state adapted spiking neurons, and this difference causes an error that may be large for fast changing signals.

Analog-valued spikes as explored in BID19 could likely resolve this issue, at the expense of some loss of representational efficiency.

Although the adaptive spiking LSTM implemented in this paper does not have output gates BID11 , they can be included by following the same approach used for the input gates: a modulation of the synaptic strength.

The reasons for our approach are multiple: first of all, most of the tasks do not really require output gates; moreover, modulating each output synapse independently is less intuitive and biologically plausible than for the input gates.

A similar argument can be made for the forget gates, which were not included in the original LSTM formulation: here, the solution consists in modulating the decaying factor of the CEC.Finally, which gates are really needed in an LSTM network is still an open question, with answers depending on the kind of task to be solved BID9 BID21 .

For example, the AuGMEnT framework does not use gates to solve many working memory RL tasks BID16 .

In addition, it has been shown by BID4 ; BID9 that a combination of input and forget gates can outperform LSTM on a variety of tasks while reducing the LSTM complexity.

@highlight

 We demonstrate a gated recurrent asynchronous spiking neural network that corresponds to an LSTM unit.