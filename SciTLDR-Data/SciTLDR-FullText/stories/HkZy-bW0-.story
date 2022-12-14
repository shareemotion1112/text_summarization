The vast majority of natural sensory data is temporally redundant.

For instance, video frames or audio samples which are sampled at nearby points in time tend to have similar values.

Typically, deep learning algorithms take no advantage of this redundancy to reduce computations.

This can be an obscene waste of energy.

We present a variant on backpropagation for neural networks in which computation scales with the rate of change of the data - not the rate at which we process the data.

We do this by implementing a form of Predictive Coding wherein neurons communicate a combination of their state, and their temporal change in state, and quantize this signal using Sigma-Delta modulation.

Intriguingly, this simple communication rule give rise to units that resemble biologically-inspired leaky integrate-and-fire neurons, and to a spike-timing-dependent weight-update similar to Spike-Timing Dependent Plasticity (STDP), a synaptic learning rule observed in the brain.

We demonstrate that on MNIST, on a temporal variant of MNIST, and on Youtube-BB, a dataset with videos in the wild, our algorithm performs about as well as a standard deep network trained with backpropagation, despite only communicating discrete values between layers.

Currently, most algorithms used in Machine Learning work under the assumption that data points are independent and identically distributed, as this assumption provides good statistical guarantees for convergence.

This is very different from the way data enters our brains.

Our eyes receive a single, never-ending stream of temporally correlated data.

We get to use this data once, and then it's gone.

Moreover, most sensors produce sequential, temporally redundant streams of data.

This can be both a blessing and a curse.

From a statistical learning point of view this redundancy may lead to biased estimators when used to train models which assume independent and identically distributed input data.

However, the temporal redundancy also implies that intuitively not all computations are necessary.

Online Learning is the study of how to learn in this domain -where data becomes available in sequential order and is given to the model only once.

Given the enormous amount of sequential data, mainly videos, that are being produced nowadays, it seems desirable to develop learning systems that simply consume data on-the-fly as it is being generated, rather than collect it into datasets for offline-training.

There is, however a problem of efficiency, which we hope to illustrate with two examples:1.

CCTV feeds.

CCTV Cameras collect an enormous amount of data from mostly-static scenes.

The amount of new information in a frame, given the previous frame, tends to be low, i.e. the data tends to be temporally redundant.

If we want to train a model from of this data (for example a pedestrian detector), we need to process a large amount of mostly-static frames.

If the frame rate doubles, so does the amount of computation.

Intuitively, it feels that this should not be necessary.

It would be nice to still be able to use all this data, but have the amount of computation scale with the amount of new information in each frame, not just the number of frames and dimensions of the data.2.

Robot perception.

Robots have no choice but to learn online -their future input data (e.g. camera frames) are dependent on their previous predictions (i.e. motor actions).

Not only does their data come in nonstationary temporal streams, but it typically comes from several sensors running at different rates.

The camera may produce 1MB images at 30 frames/s, while the gyroscope might produce 1-byte readings at 1000 frames/s.

It is not obvious, using current methods in deep learning, how we can integrate asynchronous sensory signals into a unified, trainable, latent representation, without undergoing the inefficient process of recomputing the function of the network every time a new signal arrives.

These examples point to the need for a training method where the amount of computation required to update the model scales with the amount of new information in the data, and not just the dimensionality of the data.

There has been a lot of work on increasing the computational efficiency of neural networks by quantizing neural weights or activations (see Section 4), but comparatively little work on exploiting redundancies in the data to reduce the amount of computation.

BID12 , set out to exploit the temporal redundancy in video by having neurons only send their quantized changes in activation to downstream neurons, and having the downstream neurons integrate these changes over time.

This approach (take the temporal difference, multiply by weights, temporally integrate) works for efficiently approximating the function of the network, but fails for training.

The reason for this failure is that when the weights are functions of time, we no longer reconstruct the correct activation for the next layer.

In other words, given a sequence of inputs x 0 ...x t with x 0 = 0 and weights w 1 ...w t :t ?? =1 (x ?? ??? x ?? ???1 ) ?? w ?? = x t ?? w t unless w t = w 0 ???t.

FIG0 describes the problem visually.

In this paper, we correct for this problem by encoding a mixture of two components of the layers activation x t : the proportional component k p x t , and the derivative component k d (x t ??? x t???1 ).

When we invert this encoding scheme, we get a decoding scheme which corresponds to taking an exponentially decaying temporal average of past inputs.

Interestingly, the resulting neurons begin to resemble models of biological spiking neurons, whose membrane potentials can approximately be modeled as an exponentially decaying temporal average of past inputs.

In this work, we present a scheme wherein the temporal redundancy of input data is used to reduce the computation required to train a neural network.

We demonstrate this on the MNIST and Youtube-BB datasets.

To our knowledge we are the first to create a neural network training algorithm which uses less computation as data becomes more temporally redundant.

We propose a coding scheme where neurons can represent their activations as a temporally sparse series of impulses.

The impulses from a given neuron encode a combination of the value and the rate of change of the neuron's activation.

While our algorithm is designed to work efficiently with temporal data, we do not aim to learn temporal sequences in this work.

We aim to efficiently approximate a function y t = f (x t ), where the current target y t is solely a function of the current input x t , and not previous inputs x 0 ...x t???1 .

The temporal redundancy between neighbouring inputs x t???1 , x t will however be used to make our approximate computation of this function more efficient.

Throughout this paper we will use the notation (f 3 ??? f 2 ??? f 1 )(x) = f 3 (f 2 (f 1 (x))) to denote function composition.

We slightly abuse the notion of functions by allowing them to have an internal state which persists between calls.

For example, we define the ??? function in Equation 1 as being the difference between the inputs in two consecutive calls (where persistent variable x last is initialized to 0).

The ?? function, defined in Equation 2, returns a running sum of the inputs over calls.

So we can write, for example, that when our composition of functions (?? ??? ???) is called with a sequence of input variables x 1 ... DISPLAYFORM0 In general, when we write y t = f (x t ), where f is a function with persistent state, it will be implied that we have previously called f (x ?? ) for ?? ??? [1, .., t ??? 1] in sequence.

Variable definitions that are used later will be highlighted in blue.

While all terms are defined in the paper, we encourage the reader to refer to Appendix A for a complete collection of definitions and notations.

??? :x ??? y; Persistent: DISPLAYFORM0 Q :x ??? y; Persistent: DISPLAYFORM1 enc :x ??? y; Persistent: DISPLAYFORM2 dec :x ??? y; Persistent: DISPLAYFORM3 Suppose a neuron has time-varying activation x 1 ..x t .

Taking inspiration from Proportional-Integral-Derivative (PID) controllers, we can "encode" this activation at each time step as a combination of its current activation and change in activation as a t enc( Equation 4 ).

The parameters k p and k d determine what portion of our encoding represents the value of the activation and the rate of change of that value, respectively.

In Section 4, we discuss how this is a form of Predictive Coding and in Appendix E, we discuss the effect our choices for these parameters have on the network.

DISPLAYFORM4 To derive our decoding formula, we can simply solve for x t as x t = DISPLAYFORM5 DISPLAYFORM6 We can expand this recursively to see that this corresponds to a temporal convolution a * ?? where ?? is a causal exponential kernel DISPLAYFORM7

Our motivation for the aforementioned encoding scheme is that we want a sparse signal which can be quantized into a low bitrate discrete signal.

This will later be used to reduce computation.

We can quantize our signal a t into a sparse, integer signal s t Q(a t ), where the quantizer Q is defined in Equation 3.

Equation 3 implements a form of Sigma-Delta modulation, a method widely used in signal processing to approximately communicate signals at low bit rates BID4 .

We can show that Q(x t ) = (??? ??? R ??? ??)(x t ) (see Supplementary Material Section C), where ??? ??? R ??? ?? indicates applying a temporal summation, a rounding, and a temporal difference, in series.

If x t is temporally redundant and we set k p to be small, then |a t | 1???t, and we can expect s t to consist of mostly zeros with a few 1's and -1's.

We can now approximately reconstruct our original signal x t asx t dec(s t ) by applying our decoder, as defined in Equation 5.

As our coefficients k p , k d become larger, our reconstructed signal x t should become closer to the original signal x t .

We illustrate examples of encoded signals and their reconstructions for different k p , k d in Figure 1 .

We can compactly write the entire reconstruction function asx = (dec DISPLAYFORM0 with one another, we can simplify this tox t = (k The problem with only sending changes in activation (i.e. k p = 0) is that during training, weights change over time.

Top: we generate random signals for a single scalar activation x t and scalar weight w t .

Row 2: We efficiently approximate z t by taking the temporal difference, multiplying by w t then temporally integrating, to produce??? t , as described in Section 2.4.

As the weight w t changes over time, our estimate??? diverges from the correct value.

Rows 3, 4: Introducing k p allows us to bring our reconstruction back in line with the correct signal.

DISPLAYFORM1 DISPLAYFORM2 with no dependence on x t???1 .

This is visible in the bottom row of Figure 1 .

This was the encoding scheme used in O' Connor and Welling (2016b) .

DISPLAYFORM3 p x t and enc(x t ) = k p x t so our encoding-decoding process becomesx = (k DISPLAYFORM4 .

Neither our encoder nor our decoder have any memory, and we take no advantage of temporal redundancy.

The purpose of our encoding scheme is to reduce computation by sparsifying communication between layers of a neural network.

Our approach is to approximate the matrix-product as a series of additions, where the number of additions is inversely proportional to the sparsity of the input data.

Suppose we are trying to compute the pre-nonlinearity activation of the first hidden layer, z t ??? R dout , given the input activation, x t ??? R din .

We approximate z t as: DISPLAYFORM0 where: DISPLAYFORM1 The first approximation comes from the quantization (Q) of the encoded signal, and the second from the fact that the weights change over time, as explained in FIG0 .

The effects of these approximations are further explored in Appendix E.1.

DISPLAYFORM2 The cost of computing??? t , on the other hand, depends on the contents of s t .

If the data is temporally redundant, s t ??? Z din should be sparse, with total magnitude N i |s t,i |.

s t can be decomposed into a sum of one-hot vectors DISPLAYFORM3 where ?? in is a length-d in one-hot vector with element (?? in ) in = 1.

The matrix product s t ?? w can then be decomposed into a series of row additions: DISPLAYFORM4 If we include the encoding, quantization, and decoding operations, our matrix product takes a total of 2d in + 2d out multiplications, and n |s t,n | ?? d out + 3d in + d out additions.

Assuming the n |s t,n | ?? d out term dominates, we can say that the relative cost of computing??? t vs z t is: DISPLAYFORM5 2.5 A NEURAL NETWORKWe can implement this encoding scheme on every layer of a neural network.

Given a standard neural net f nn consisting of alternating linear (??w l ) and nonlinear (h l ) operations, our network function f pdnn can then be written as: DISPLAYFORM6 We can use the same approach to approximately calculate our gradients to use in training.

If we define our layer activations as??? l (dec DISPLAYFORM7 , where is some loss function and y is a target, we can backpropagate the approximate gradients as: DISPLAYFORM8 This can be implemented by either executing a (sparse) forward and backward pass at each time-step, or in an "event-based" manner, where the quantizers fire "events" whenever incoming events push their activations past a threshold, and these events are in turn sent to downstream neurons.

For ease of implementation, we opt for the former in our code.

Note that unlike in regular backprop, computing these forward and backward passes results in changes to the internal state of the enc, dec, and Q components.

There is no use in having an efficient backward pass if the parameter updates are not also efficient.

In a normal neural network trained with backpropagation and simple stochastic gradient descent, the DISPLAYFORM0 , and its reconstructionx t = dec(x t ).

Middle: Another signal, representing the postsynaptic gradient of the error e = ???L ???z l , along with its quantized (??) and reconstructed (??) variants.

Bottom:The true weight gradient ???L ???wt and the reconstruction gradient ???L ???wt .

At the time of the spike in?? t , we have two schemes for efficiently computing the weight gradient that will be used to increment weight (see Section 2.6).

The past scheme computes the area underx ???? since the last spike, and the future scheme computes the total future additional area due to the current spike.parameter update for weight matrix w has the form w ??? w ??? ?? ???L ???w where ?? is the learning rate.

If w connects layer l ??? 1 to layer l, we can write DISPLAYFORM1 ??? R dout is the postsynaptic (layer l) backpropagating gradient and ??? is the outer product.

So we require d in ?? d out multiplications to update the parameters for each sample.

We want a more efficient way to compute this product, which takes advantage of the sparsity of our encoded signals to reduce computation.

We can start by applying our encoding-quantizing-decoding scheme to our input and error signals asx DISPLAYFORM2 and approximate our true update as ???L ???w recon,tx t ????? t wherex t dec(x t ) and?? t dec(?? t ).

This does not do any good by itself, because the vectors involved in the outer product,x t and?? t , are still not sparse.

However, we can exactly compute the sum of this value over time using one of two sparse update schemes -past updates and future updates -which are depicted in FIG1 .

We give the formula for the Past and Future update rules in Appendix D, but summarize them here:Past Updates:

For a given synapse w i,j , if either the presynaptic neuron spikes (x ti = 0) or the postsynaptic neuron spikes (?? ti = 0), we increment the w i,j by the total area underx ??,i????,j since the last spike.

We can do this efficiently because between the current time and the time of the previous spike,x ??,i????,j is a geometric sequence.

Given a known initial value u, final value v, and decay rate r, a geometric sequence sums to Future Updates: Another approach is to calculate the Present Value of the future area under the integral from the current spike.

This is depicted in the blue-gray area in FIG1 , and the formula is in Equation 20 in Appendix D.Finally, because the magnitude of our gradient varies greatly over training, we create a scheme for adaptively tuning our k p , k d parameters to match the running average of the magnitude of the data.

This is described in detail in Appendix E.

An extremely attentive reader might have noted that Equation 20 has the form of an online implementation of Spike-Timing Dependent Plasticity (STDP).

STDP BID10 Using the quantized input signalx and error signal??, and their reconstructionsx t and?? t as defined in the last section, we define a causal convolutional kernel DISPLAYFORM0 .

We can then define a "cross-correlation kernel" DISPLAYFORM1 |t| : t ??? Z which defines the magnitude of a parameter update as a function of the difference in timing between pre-synaptic spikes from the forward pass and post-synaptic spikes from the backward pass.

The middle plot of FIG3 is a plot of g.

We define our STDP update rule as: DISPLAYFORM2 We note that while our version of STDP has the same double-exponential form as the classic STDP rule observed in neuroscience BID10 , our "presynaptic" spikes come from the forward pass while our "postsynaptic" spikes come from the backwards pass.

STDP is not normally used to as a learning rule networks trained by backpropagation, so the notion of forward and backward pass with a spike-timing-based learning rule are new.

Moreover, unlike in classic STDP, we do not have the property that sign of the weight change depends on whether the presynaptic spike preceded the postsynaptic spike.

In Section D in the supplementary material we show experimentally that while Equations may all result in different updates at different times, the rules are equivalent in that for a given set of pre/post-synaptic spikesx,??, the cumulative sum of their updates over time converges exactly.

To evaluate our network's ability to learn, we train it on the standard MNIST dataset, as well as a variant we created called "Temporal MNIST".

Temporal MNIST is simply a reshuffling of the MNIST dataset so that so that similar inputs (in terms of L2-pixel distance), are put together.

FIG4 shows several snippets of consecutive frames in the temporal MNIST dataset.

We compare our Proportional-Derivative Net against a conventional Multi-Layer Perceptron with the same architecture (one hidden layer of 200 ReLU hidden units and a softmax output).

The results are shown in Figure 5 .

Somewhat surprisingly, our predictor slightly outperformed the MLP, getting 98.36% on the test set vs 98.25% for the MLP.

We assume this improvement is due to the regularizing effect of the quantization.

On Temporal MNIST, our network was able to converge with less computation than it required for MNIST (it used 32 ?? 10 12 operations for MNIST vs 15 ?? 10 12 for Temporal MNIST), but FORMULA1 for the computational costs of arithmethic operations (0.1pJ for 32-bit fixed-point addition vs 3.2pJ for multiplication), we can see that our algorithm would be at an advantage on any hardware where arithmetic operations were the computational bottleneck.

ended up with a slightly worse test score when compared with the MLP (the PDNN achieved 97.99% vs 98.28% for the MLP).

The slightly higher performance of the MLP on Temporal MNIST may be explained by the fact that the gradients on Temporal MNIST tend to be correlated across time-steps, so weights will tend to move in a single direction for a number of steps, which will interfere with the PDNN's ability to accurately track layer activations (see FIG0 .

Appendix F contains a table of results with varying hyperparameters.

Next, we want to simulate the setting of CCTV cameras, discussed in Section 1, where we have a lot of data with only a small amount of new information per frame.

In the absence of large enough public CCTV video datasets, we investigate the surrogate task of frame-based object classification on wild YouTube videos from the large, recently released Youtube-BB dataset BID13 .

Our subset consists of 358 Training Videos and 89 Test videos with 758,033 frames in total.

Each video is labeled with an object in one of 24 categories.

We start from a VGG19 network BID15 : a 19-layer convolutional network pre-trained on imagenet.

We replace the top three layer with three of our own randomly initialized layers, and train the network both as a spiking network, and as a regular network with backpropagation.

While training the entire spiking network end-to-end works, we choose to only train the top layers, in order to speed up our training time.

We compare our training scores and computation between a spiking and non-spiking implementation.

The learning curves in FIG5 show that our spiking network performs comparably to a non-spiking network, and Figure 8 shows how the computation per frame of our spiking network decreases as we increase the frame rate (i.e. as the input data becomes more temporally redundant).

Because our spiking network uses only additions, while a regular deep network does multiply-adds, we use the estimated energy-costs per op of BID8 to compare computations to a single scale, which estimates the amount of energy required to do multiplies and adds in fixed-point arithmetic.

Figure 8 : We simulate different frame-rates by selecting every n'th frame.

This plot shows our network's mean computation over several snippets of video, at varying frame rates.

As our frame rate increases, the computation per-frame of our spiking network goes down, while with a normal network, it remains fixed.

Noise-Shaping is a quantization technique that aims to increase the fidelity of signal reconstructions, per unit of bandwidth of the encoded signal, by quantizing the signal in such a way that the quantization noise is pushed into a higher frequency band which is later filtered out upon decoding.

Sigma-Delta (also known as Delta-Sigma) quantization is a form of noise-shaping.

BID14 first suggested that biological neurons may be performing a form of noise shaping, and Yoon (2017) found standard spiking neuron models actually implement a form of Sigma-Delta modulation.

The encoding/decoding scheme we use in this paper can be seen as a form of Predictive Coding.

Predictive coding is a lossless compression technique wherein the predictable parts of a signal are subtracted away so that just the unpredictable parts are transmitted.

The idea that biological neurons may be doing some form of predictive coding was first proposed by BID16 .

In a predictive-coding neuron (unlike neurons commonly used in Deep Learning), there is a distinction between the signal that a neuron represents and the signal that it transmits.

The neurons we use in this paper can be seen as implementing a simple form of predictive coding where the "prediction" is that the neuron maintains a decayed form of its previous signal -i.e.

that pred(x t ) DISPLAYFORM0 x t???1 (See Appendix B for detail).

BID5 suggest that the biological spiking mechanism may be thought of as consisting of a sigma-delta modulator within a predictive-coding circuit.

To our knowledge, none of the aforementioned work has yet been used in the context of deep learning.

There has been sparse but interesting work on merging the notions of spiking neural networks and deep learning.

BID7 found a way to efficiently map a trained neural network onto a spiking network.

BID9 devised a method for training integrate-and-fire spiking neurons with backpropagation -though their neurons did not send a temporal difference of their activations.

O'Connor and Welling (2016a) created a method for training event-based neural networks -but their method took no advantage of temporal redundancy in the data.

BID2 and (O'Connor and Welling, 2016b) both took the approach of sending quantized temporal changes to reduce computation on temporally redundant data, but their schemes could not be used to train a neural network.

BID3 showed how one could apply backpropagation for training spiking neural networks, but it was not obvious how to apply the method to non-spiking data.

BID19 developed a spiking network with an adaptive scale of quantization (which bears some resemblance to our tuning scheme described in Appendix E), and show that the spiking mechanism is a form of Sigma-Delta modulation, which we also use here.

BID6 showed that neural networks could be trained with binary weights and activations (we just quantize activations).

found a connection between the classic STDP rule FIG3 , right) and optimizing a dynamical neural network, although the way they arrived at an STDP-like rule was quite different from ours (they frame STDP as a way to minimze an objective based on the rate of change of the real-valued state of the network, whereas we use it approximately compute gradients based on spike-encodings of layer activations).

We set out with the objective of reducing the computation in deep networks by taking advantage of temporal redundancy in data.

We described a simple rule (Equation 4) for sparsifying the communication between layers of a neural network by having our neurons communicate a combination of their temporal change in activation, and the current value of their activation.

We show that it follows from this scheme that neurons should behave as leaky integrators (Equation 5 ).

When we quantize our neural activations with Sigma-Delta modulation, a common quantization scheme in signal processing, we get something resembling a leaky integrate-and-fire neuron.

We derive efficient update rules for the weights of our network, and show these to be related to STDP -a learning rule first observed in neuroscience.

Finally, we train our network, verify that it does indeed compute more efficiently on temporal data, and show that it performs about as well as a traditional deep network of the same architecture, but with significantly reduced computation.

Finally, we show that our network can train on real video data.

The efficiency of our approach hinges on the temporal redundancy of our input data and neural activations.

There is an interesting synergy here with the concept of slow-features BID17 .

Slow-Feature learning aims to discover latent objects that persist over time.

If the hidden units were to specifically learn to respond to slowly-varying features of the input, the layers in a spiking implementation of such a network would have to communicate less often.

In such a network, the tasks of feature-learning and reducing inter-layer communication may be one and the same.

Code is available at github.com/petered/pdnn.

This work was supported by Qualcomm, who we'd also like to thank for sharing their past work with us.

Here we present a legend of notation used throughout this paper.

While the paper is intended to be self-contained, the reader may want to consult this list if ever there is any doubt about the meaning of a variable used in the paper.

Here we indicate the section in which each symbol is first used.

??: The internal state variable of the quantizer Q. enc: An "encoding" operation, which takes a signal and encodes it into a combination of the signal's current value and its change in value since the last time step.

See Equation 4.

dec: A "decoding" operation, which takes an encoding signal and attempts to reconstruct the original signal that it was encoded from.

If there was no quantization done on the encoded signal, the reconstruction will be exact, otherwise it will be an approximation.

See Equation 5.

The "rounding" operation, which simply rounds an input to the nearest integer value.x t : Used throughout the paper to represent the value of a generic analog input signal at time t. In Sections 2.1, 2.2, and 2.3 it represents a scalar, and thereafter it represents a vector of inputs.

DISPLAYFORM0 Positive scalar coefficients used in the encoder and decoder, controlling how the extent to which the encoding is proportional to the input (k p ) vs proportional to the temporal difference of the input (k d ).a t enc(x t ): Used to represent the encoded version of x t .

Section 2.3 s t Q(a t ): Used to represent the quantized, encoded version of x t .x t dec(s t ): Used to represent the reconstruction of input x t " obtained by encoding, quantizing, and decoding x t .

Section 2.4 w t ??? R din??dout is the value of a weight matrix at time t.z t x t ?? w t ??? R dout is the value of a pre-nonlinearity hidden layer activation in a non-spiking network at time t. DISPLAYFORM1 dout is the value of a pre-nonlinearity hidden layer activation in the spiking network at time t.

It is an approximation of z t .

(??w l ) indicates applying a function which takes the dot-product of the input with the l'th weight matrix: (??w l )(x) x ?? w l h l indicates an elementwise nonlinearity (e.g. a ReLU).Q l indicates the quantization step applied at the l'th layer (because quantization has internal state, ??, and an associated layer dimension, we use the subscript to distinguish quantizers at different layers.) dec l , enc l are likewise the (stateful) encoding/decoding functions applied before/after layer l. z l is the approximation to the (pre-nonlinearity) activation to layer l (ie the output of dec l ), computed by the spiking network.( h l (??? l )) is a function that performs an elementwise multiplication of the input with the derivative of nonlinearity h l evaluated at??? l .

This is simply backpropagation across a nonlinearity: If u h l (x), then DISPLAYFORM0 serve the same functions as dec l , enc l , Q l , but for the backward pass.

???L ?????? l ??? R d l Is our approximation to the derivative of the loss of our network with respect to??? l , which is itself an approximation of the activation z l in a non-spiking implementation of the network.

In the updates section we describe how we calculate the weight gradients in layer l. Because this description holds for any arbitrary layer, we get rid of the layer subscript and use the following notation: DISPLAYFORM0 here is defined as a shorthand for "the input to layer l".

e t ???L ?????? l,t ??? R dout is simply a shorthand for "the approximate backpropagated gradient at layer l" (x t and?? t ) are the encoded and quantized versions of signals (x t and e t ) (x t and?? t ) are the reconstructed versions of signals (x t and e t ), taken from the quantized (x t and?? t ) ???L ???w recon,tx t ????? t ??? R din??dout is the approximate gradient of weight matrix w, as calculated by taking the outer product of the (input, error) reconstructions,x,??. (see Figure 9 ).

???L ???w stdp,t is the gradient approximation taken using the STDP-type update.

It also converges to the same value as ???L ???w recon,t when averaged out over time.

DISPLAYFORM0 A reparametrization of k p , k d in terms of the memory in our decoder k ?? and the scaling of our encoded signal (k ?? ).

This reparametrization is also used when discussing the automatic tuning of k p , k d to match the dynamic range of our data in Appendix E B RELATION TO PREDICTIVE CODING Our encoding/decoding scheme is an instance of predictive coding -an idea imported from the signal processing literature into neuroscience by BID16 wherein the power of a transmitted signal is reduced by subtracting away the predictable component of this signal before transmission, then reconstructing it after (This requires that the encoder and decoder share the same prediction model).

BID1 formulate feedforward predictive coding as follows (with variables names changed to match the conventions of this paper): DISPLAYFORM1 In the case of Linear Predictive Coding (15) Where the reconstruction is done by: DISPLAYFORM2 They go on to define "optimal" liner filter parameters [w 1 , w 2 , ...] that minimize the average magnitude of a t in terms of the autocorrelation and signal-to-noise ratio of x.

Our scheme defines: DISPLAYFORM3 So it is identical to feedforward predictive coding with DISPLAYFORM4 In our case, the function of this additional constant is to determine the coarseness of the quantization.

From this relationship it is clear that this work could be extended to come up with more efficient predictive coding schemes which could further reduce computation by learning the temporal characteristics of the input signal.

C SIGMA-DELTA UNWRAPPINGHere we show that Q = ??? ??? R ??? ??, where Q, ???, R, ?? are defined in Equations 3, 2, 6, 1, respectively.

From Equation 3 (Q) we can see that DISPLAYFORM5 Now we can unroll for y t and use the fact that if s ??? Z then round(a + s) = round(a) + s: DISPLAYFORM6 At which point it is clear that Q is identical to a successive application of a temporal summation, a rounding, and a temporal difference.

That is why we say Q = ??? ??? R ??? ??.

In Section 2.6, we visually describe what we call the "Past" and "Future" parameters updates.

Here we present the algorithms for implementing these schemes.

To simplify our expressions in the update algorithms, we re-parametrize our k p , k d coefficients as DISPLAYFORM0 Equation FORMULA15 shows how we make two approximations when approximating z t = x t ?? w t with??? t = (dec ??? w ??? Q ??? enc)(x t ).

The first is the "nonstationary weight" approximation, arising from the fact that w changes in time.

The second is the "quantization" approximation, arising from the quantization of x.

Here we do a small experiment in which we multiply a time-varying scalar signal x t with a time-varying weight w t for many different values of k p , k d to understand the effects of k p , k d on our approximation error.

The bottom-middle plot in Figure 10 shows that we enter a high-reconstructionerror regime (blue on plot) when k d is small (high quantization error), or when k d >> k p (high nonstationary-weight error).

The bottom-right plot shows that blindly increasing k p and k d leads to representing the signal with many more spikes.

Thus we need to tune hyperparameters to find the "sweet spot" where reconstruction error is fairly low but our encoded signal remains fairly sparse, keeping computational costs low.

Figure 10: Top Left: A time varying signal x t , the quantized signal Q(enc(x t )), and the time-varying "weight" w t .

Bottom Left: Compare the true product of these signals x t ??w t with the dec(enc(x t )??w t ), which shows the effects of the non-stationary weight approximation, and dec(Q(enc(x t )) ?? w) which shows both approximations.

Top Middle: The Cosine distance between the "true" signal x w and the approximation due to the nonstationary w, scanned over a grid of DISPLAYFORM1 The cosine distance between the "true" signal and the approximation due to the quantization of x. Bottom Middle: The Cosine Distance between the "true" signal and the full approximation described in Equation 7.

This shows why we need both k p and k d to be nonzero.

Bottom Right: The Number of spikes in the encoded signal.

In a neural network this would correspond to the number of weight-lookups required to compute the next layer's activation: dec(Q(enc(x)) w).

DISPLAYFORM2 The smaller the magnitude of a signal, the more severely distorted it is by our quantizationreconstruction scheme.

We can see that scaling a signal by K has the same effect on the quantized version of the signal, s t , as scaling k p and k d by K: DISPLAYFORM3 The fact that the reconstruction quality depends on the signal magnitude presents a problem when training our network, because the error gradients tend to change in magnitude throughout training (they start large, and become smaller as the network learns).

To keep our signal within the useful dynamic range of the quantizer, we apply a simple scheme to heuristically adjust k p and k d for the forward and backward passes separately, for each layer of the network.

Instead of directly setting k p , k d as hyperparameters, we fix the ratio k ?? DISPLAYFORM4 , and adapt the scale k ?? 1 kp+k d to the magnitude of the signal.

Our update rule for k ?? is: DISPLAYFORM5 Where ?? k is the scale-adaptation learning rate, ?? t is a rolling average of the L 1 magnitude of signal x t , and k rel ?? defines how coarse our quantization should be relative to the signal magnitude (higher means coarser).

We can recover k p , k d for use in the encoders and decoders as k p = (1 ??? k ?? )/k ?? and k d = k ?? /k ?? .

In our experiments, we choose ?? k = 0.001, k rel ?? = 0.91, k alpha = 0.91, and initialize ?? 0 = 1.

Here we show training scores and computation for the PDNN and MLP under different inputorderings (the unordered MNIST vs the ordered Temporal MNIST) and hidden layer depths.

We notice no dropoff in performance of the PDNN (as compared to an MLP) with the same architecture as we add hidden layers -indicating that the accumulation of quantization noise over layers appears not to be a problem.

For all experiments, the PDNN started with k ?? = 0.5, and this was increased to k ?? = 0.9 after 1 epoch (see Appendix A for the meaning of k ?? ).

Note that the numbers for Mean Computation are counting additions for the PDNN, and multiply-adds for the MLP, so they are not directly comparable (a 32-bit multiply, if implemented in fixed point, is 32 times more energetically expensive than an add BID8 Figure 11 : 16 Frames from the Youtube-BB dataset.

Each video annotated as having one of 24 objects in it.

It also comes with annotated bounding-boxes, which we do not use in this study.

Figure 8 seems to show that computation doesn't quite approach zero as our frame-rate increases, but flat-lines at a certain point.

We think this may have to do with the fact that hidden layer activations are not necessarily smoother in time than the input.

We demonstrate this by taking 5 video snippets from the Youtube-BB dataset and running them through a (non-spiking) 19 layer VGGNet architectures BID15 , which was pre-trained on ImageNet.

Given these 5 snippets, we measures how much the average relative change in layer activation |at???at???1| 2(|at|+|at???1|) varies as we increase our frame-rate, at various layer-depths.

We simulate lower frame rates by skipping every N'th frame of video. (so for example to get a 10FPS frame rate we simply select every 3rd frame of the 30FPS video).

For each selected frame rate, and for the given layers, we measure the average inter-frame change at various layers:F P S(n) = 30/n x-axis FORMULA1 RelChange(n) = 1 S S s=1 T /n t=1 |a nt ??? a (n???1)t | 2(|a nt | + |a (n???1)t |) y-axisWhere: S = 5 is the number of video snippets we average over T is the number of frames in each snippet a t is the activation of a layer at time t n is the number of frames we are skipping over.

This shows something interesting.

While our deeper layers do indeed show less relative change in activation over frames than our input/shallow layers, we note that as frame-rate increases, this seems to approach zero much more slowly than our input/shallow layers.

This is a problem for our method, which relies on temporal smoothness in all layers (especially those hidden layers with large amounts of downstream units) to save computation.

It suggests that methods for learning slow feature detectors -layers that are trained specifically to look for slowly varying features of the input, may be helpful to us.

FIG0 : The average relative change in layer activation between frames, as frame-rate increases.

For increasing network depth (red=shallow, violet=deep)

@highlight

An algorithm for training neural networks efficiently on temporally redundant data.

@highlight

The paper describes a neural coding scheme for spike based learning in deep neural networks

@highlight


This paper presents a method for spike based learning that aims at reducing the needed computation during learning and testing when classifying temporal redundant data.

@highlight

This paper applies a predictive coding version of the Sigma-Delta encoding scheme to reduce a computational load on a deep learning network, combining the three components in a way not seen previously.