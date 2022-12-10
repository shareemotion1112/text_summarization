In vanilla backpropagation (VBP), activation function matters considerably in terms of non-linearity and differentiability.

Vanishing gradient has been an important problem related to the bad choice of activation function in deep learning (DL).

This work shows that a differentiable activation function is not necessary any more for error backpropagation.

The derivative of the activation function can be replaced by an iterative temporal differencing (ITD) using fixed random feedback weight alignment (FBA).

Using FBA with ITD, we can transform the VBP into a more biologically plausible approach for learning deep neural network architectures.

We don't claim that ITD works completely the same as the spike-time dependent plasticity (STDP) in our brain but this work can be a step toward the integration of STDP-based error backpropagation in deep learning.

VBP was proposed around 1987 BID10 .

Almost at the same time, biologicallyinspired convolutional networks was also introduced as well using VBP BID5 .

Deep learning (DL) was introduced as an approach to learn deep neural network architecture using VBP BID5 ; BID4 .

Extremely deep networks learning reached 152 layers of representation with residual and highway networks BID3 ; BID13 .

Deep reinforcement learning was successfully implemented and applied which was mimicking the dopamine effect in our brain for self-supervised and unsupervised learning BID11 BID9 BID8 .

Hierarchical convolutional neural network have been biologically inspired by our visual cortex Hubel & Wiesel (1959) ; BID1 BID0 BID14 .

The discovery of fixed random synaptic feedback weights alignments (FBA) in error backpropagation for deep learning started a new quest of finding the biological version of VBP since it solves the symmetrical synaptic weights problem in backprop.

Recently, spiketime dependent plasticity was the important issue with backprop.

One of the works in this direction, highly inspired from Hinton's recirculation idea Hinton & McClelland (1988) , is deep learning using segregated dendrites BID2 .

Apical dendrites as the segregated synaptic feedback are claimed to be capable of modeling STDP into the backprop successfully BID2 .

In this section, we visually demonstrate the ITD using FBA in VBP 1.

In this figure, VBP, VBP with FBA, and ITD using FBA for VBP are shown all in one figure.

The choice of activation function for this implementation was Tanh function.

The ITD was applied to MNIST standard dataset.

VBP, FBA, and ITD were compared using maximum cross entropy (MCE) as the loss function 2.

Also, ITD with MCE as loss function is compared to ITD with least squared error (LSE) 3.

The hyper parameters for both of the experiments are equal as follows: 5000 number of iterations/ epochs, 0.01 (1e-2) learning rate, 100 minibatch size with shuffling for stochasticity, vanilla stochastic gradient descent is used, 32 for number of hidden layers, 2-layer deep networks.

Feed-forward neural network is used as the architecture.

In this paper, we took one more step toward a more biologically plausible backpropagation for deep learning.

After hierarchical convolutional neural network and fixed random synaptic feedback alignment, we believe iterative temporal differencing is a way toward integrating STDP learning process in the brain.

We believe the next steps should be to investigate more into the STDP processes details in learning, dopamine-based unsupervised learning, and generating Poisson-based spikes.

@highlight

Iterative temporal differencing with fixed random feedback alignment support spike-time dependent plasticity in vanilla backpropagation for deep learning.