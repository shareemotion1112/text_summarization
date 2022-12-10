Recurrent Neural Networks (RNNs) are widely used models for sequence data.

Just like for feedforward networks, it has become common to build "deep" RNNs, i.e., stack multiple recurrent layers to obtain higher-level abstractions of the data.

However, this works only for a handful of layers.

Unlike feedforward networks, stacking more than a few recurrent units (e.g., LSTM cells) usually hurts model performance, the reason being vanishing or exploding gradients during training.

We investigate the training of multi-layer RNNs and examine the magnitude of the gradients as they propagate through the network.

We show that, depending on the structure of the basic recurrent unit, the gradients are systematically attenuated or amplified, so that with an increasing depth they tend to vanish, respectively explode.

Based on our analysis we design a new type of gated cell that better preserves gradient magnitude, and therefore makes it possible to train deeper RNNs.

We experimentally validate our design with five different sequence modelling tasks on three different datasets.

The proposed stackable recurrent (STAR) cell allows for substantially deeper recurrent architectures, with improved performance.

Recurrent Neural Networks (RNN) have established themselves as a powerful tool for modelling sequential data.

They have significantly advanced a number of applications, notably language processing and speech recognition (Sutskever et al., 2014; Graves et al., 2013; Vinyals & Le, 2015) .

The basic building block of an RNN is a computational unit (or cell) that combines two inputs: the data of the current time step in the sequence and the unit's own output from the previous time step.

While RNNs are an effective approach that can in principle handle sequences of arbitrary and varying length, they are (in their basic form) challenged by long-term dependencies, since learning those would require the propagation of gradients over many time steps.

To alleviate this limitation, gated architectures have been proposed, most prominently Long Short-Term Memory (LSTM) cells (Hochreiter & Schmidhuber, 1997) and Gated Recurrent Units (GRU, Chung et al., 2014) .

They use a gating mechanism to store and propagate information over longer time intervals, thus mitigating the vanishing gradient problem.

Although such networks can, in principle, capture long-term dependencies, it is known that more abstract and longer-term features are often represented better by deeper architectures (Bengio et al., 2009) .

To that end, multiple recurrent cells are stacked on top of each other in a feedforward manner, i.e., the output (or the hidden state) of the lower cell is connected to the input gate of the next-higher cell.

Many works have used such deep recurrent architectures, e.g., (Chung et al., 2015; Zilly et al., 2017) , and have shown their ability to extract more complex features from the input and make better predictions.

The need for multi-layer RNNs is particularly apparent for image-like input data, where multiple convolutional layers are required to extract a good representation, while the recurrence captures the evolution of each layer over time.

Since recurrent architectures are trained by propagating gradients across time, it is convenient to "unwrap" them into a lattice with two axes for depth (abstraction level) and time, see Fig. 1 .

This view makes it apparent that gradients flow in two directions, namely backwards in time and downwards from deeper to shallower layers.

In this paper we ask the question how the basic recurrent unit must be designed to ensure the "vertical" gradient flow across layers is stable and not impaired by vanishing or exploding gradients.

We show that stacking several layers of common RNN cells, by their construction, leads to instabilities (e.g., for deep LSTMs the gradients tend to vanish; for deep vanilla RNNs they tend to explode).

Our study makes three contributions: (i) We analyse how the magnitude of the gradient changes as it propagates through a cell of the two-dimensional deep RNN lattice.

We show that, depending on the inner architecture of the employed RNN cell, gradients tend to be either amplified or attenuated.

As the depth increases, the repeated amplification (resp., attenuation) increases the risk of exploding (resp., vanishing) gradients. (ii) We then leverage our analysis to design a new form of gated cell, termed the STAR (stackable recurrent) unit, which better preserves the gradient magnitude inside the RNN lattice.

It can therefore be stacked to much greater depth and still remains trainable. (iii) Finally, we compare deep recurrent architectures built from different basic cells in an extensive set of experiments with three popular datasets.

The results confirm our analysis: training deep recurrent nets fail with most conventional units, whereas the proposed STAR unit allows for significantly deeper architectures.

In several cases, the ability to go deeper also leads to improved performance.

Vanishing or exploding gradients during training are a long-standing problem of recurrent (and other) neural networks (Hochreiter, 1991; Bengio et al., 1994) .

Perhaps the most effective measure to address them so far has been to introduce gating mechanisms in the RNN structure, as first proposed by (Hochreiter & Schmidhuber, 1997) in the form of the LSTM (long short-term memory), and later by other architectures such as gated recurrent units (GRU, Chung et al., 2014) .

Importantly, RNN training needs proper initialisation.

and (Henaff et al., 2016) have shown that initializing the weight matrices with identity and orthogonal matrices can be useful to stabilise the training. (Arjovsky et al., 2016) and (Wisdom et al., 2016) further develop this idea and impose orthogonality throughout the entire training to keep the amplification factor of the weight matrices close to unity, leading to a more stable gradient flow.

Unfortunately, it has been shown (Vorontsov et al., 2017 ) that such hard orthogonality constraints hurt the representation power of the model and in some cases even destabilise the optimisation.

Another line of work has studied ways to mitigate the vanishing gradient problem by introducing additional (skip) connections across time and/or layers. (Campos et al., 2018) have shown that skipping state updates in RNN shrinks the effective computation graph and thereby helps to learn longer-range dependencies. (Kim et al., 2017) introduce a residual connection between LSTM layers. (Chung et al., 2015) propose a gated feedback RNN that extends the stacked RNN architecture with extra connections.

An obvious disadvantage of such an architecture are the extra computation and memory costs of the additional connections.

Moreover, the authors only report results for rather shallow networks up to 3 layers.

Despite the described efforts, it remains challenging to train deep RNNs. (Zilly et al., 2017) have proposed Recurrent Highway Networks (RHN) that combine LSTMs and highway networks (Srivastava et al., 2015) to train deeper architectures.

RHN are popular and perform well on language modelling tasks, but they are still prone to exploding gradients, as illustrated in our experiments. (Li et al., 2018a) propose a restricted RNN where all interactions are removed between neurons in the hidden state of a layer.

This appears to greatly reduce the exploding gradient problem (allowing up to 21 layers), at the cost of a much lower representation power per layer.

To process image sequence data, computer vision systems often rely on Convolutional LSTMs (convLSTM, Xingjian et al., 2015) .

But while very deep CNNs are very effective and now standard (Krizhevsky et al., 2012; Simonyan & Zisserman, 2015) , stacks of more than a few convLSTMs do not train well.

In practice, shallow versions are preferred, for instance (Li et al., 2018b ) use a single layer for action recognition, and (Zhang et al., 2018) use two layers to recognize for hand gestures (combined with a deeper feature extractor without recursion).

We note that attempts to construct a deep counterpart to the Kalman filter can also be interpreted as recurrent networks (Krishnan et al., 2015; Becker et al., 2019; Coskun et al., 2017) .

These provide a probabilistic, generative perspective on RNNs, but are even more complicated to train.

It is at this point unclear how the basic units of these architectures could be stacked into a deep, multi-layer representation.

A RNN cell is a non-linear transformation that maps the input signal x t at time t and the hidden state of the previous time step t − 1 to the current hidden state h t :

with W the trainable parameters of the cell.

The input sequences have an overall length of T , which can be variable.

It depends on the task whether the relevant target prediction, for which also the loss L should be computed, is the final state h T , the complete sequence of states {h t }, or a single sequence label, typically defined as the average When stacking multiple RNN cells on top of each other, one passes the hidden state of the lower level l − 1 as input to the next-higher level l, see Fig. 1 , which in mathematical terms corresponds to the recurrence relation h

(2) Temporal unfolding leads to a two-dimensional lattice with depth L and length T , as in Fig. 1 .

In this computation diagram, the forward pass runs from left to right and from bottom to top.

Gradients flow in opposite direction: at each cell the gradient w.r.t.

the loss arrives at the output gate and is used to compute the gradient w.r.t.

(i) the weights, (ii) the input, and (iii) the previous hidden state.

The latter two gradients are then propagated through the respective gates to the preceding cells in time and depth.

In the following, we investigate how the magnitude of these gradients changes across the lattice.

The analysis, backed up by numerical simulations, shows that common RNN cells are biased towards attenuating or amplifying the gradients, and thus prone to destabilising the training of deep recurrent networks.

At a single cell in the lattice the gradient w.r.t.

the trainable weights are

where ∂h l t ∂w denotes the Jacobian matrix and g h l t is a column vector containing the partial derivatives of the loss w.r.t.

the cell's output (hidden) state.

From the equation, it becomes clear that the Jacobian acts as a "gain matrix" on the gradients, and should on average preserve their magnitude to prevent them from vanishing or exploding.

By expanding the gradient g h l t we obtain the recurrence for propagation,

from which we get the two Jacobians

where D x denotes a diagonal matrix with the elements of vector x as diagonal entries.

Ideally, we would like to know the expected values of the two matrices' singular values.

Unfortunately, there is no easy way to derive closed-form analytical expressions for them, but we can compute them for a fixed, representative point.

Perhaps the most natural and illustrative choice is to set h l−1 t = h l t−1 = b = 0, and to further choose orthogonal weight matrices W h and W x , a popular initialisation strategy.

Since the derivative tanh (0) = 1, the singular values of all matrices in Eq. (7) are equal to 1 in this configuration.

and g h l t+1

we expect to obtain a gradient g h l

) over all time steps.

As the gradients flow back through time and layers, for a network of vanilla RNN units they get amplified; for LSTM units they get attenuated; whereas the proposed STAR unit approximately preserves their magnitude.

in the time direction, but cannot help the flow through the layers.

Again the numerical simulation results support our hypothesis, as can be seen in Fig. 2 .

The LSTM gradients propagate relatively well backward through time, but vanish quickly towards shallower layers.

We refer to the appendix for further numerical analysis, e.g., LSTMs with only a forget gate, and GRUs.

Here, we briefly draw some connections between our analysis and the empirical results of Chung et al. (2015) , who propose a gated feedback RNN (GFRNN) that extends the stacked RNN architecture with extra connections between adjacent layers.

Empirically, GFRNN improves a 3-layer LSTM, but degrades the vanilla RNN performance.

We conjecture that this might be due to the extra connections strengthening the gradient propagation.

According to our findings, the additional gradient flow would benefit the LSTM, by bolstering the dwindling gradients; whereas for the vRNN, where the initial gradients are already too high, the added flow might be counterproductive.

Building on the analysis above, we introduce a novel RNN cell designed to avoid vanishing or exploding gradients as much as possible.

We start from the Jacobian matrix of the LSTM cell and examine in more detail which design features are responsible for the low singular values.

In equation 8 we see that every multiplication with tanh non-linearities (D tanh(.) ), gating functions (D σ(.) ), and with their derivatives can only ever decrease the singular values of W , since all this terms are always <1.

The effect is particularly pronounced for the sigmoid and its derivative, |σ (·)| ≤ 0.25 and E[|σ (x)|] = 0.5 for zero-mean, symmetric distribution of x. In particular, the output gate o l t is a sigmoid and plays a major role in shrinking the overall gradients, as it multiplicatively affects all parts of both Jacobians.

As a first measure, we thus propose to remove the output gate.

A secondary consequence of this measure is that now h l t and c l t carry the same information (the hidden state becomes an element-wise non-linear transformation of the cell state).

To avoid this duplication and further simplify the design, we transfer the tanh non-linearity to the hidden state and remove the cell state altogether.

As a final modification, we also remove the input gate i l t from the architecture.

We have empirically observed that the presence of the input gate does not significantly improve performance, moreover, it actually harms the training for deeper networks.

This empirical observation is in line with the results of van der Westhuizen & Lasenby (2018) , who show that removing the input and output gates does not greatly affect the performance of LSTMs.

More formally, our proposed STAR cell in the l-th layer takes the input h l−1 t (in the first layer, x t ) at time t and non-linearly projects it to the space where the hidden vector h l lives, equation 10.

Furthermore, the previous hidden state and the new input are combined into the gating variable k l t (equation 11).

k l t is our analogue of the forget gate and controls how the information from previous hidden state and the new input are fused into a new hidden state.

One could also intuitively interpret k l t as a sort of "Kalman gain": if it is large, the new observation is deemed reliable and dominates; otherwise the previous hidden state is conserved.

The complete dynamics of the STAR unit is given by the expressions

These equations lead to the following Jacobian matrices:

Coming back to our previous analysis for state zero and orthogonal weight matrices, each of the two Jacobians now has singular values equal to 0.5.

I.e., they lie between the vRNN cell and the LSTM cell, and when added together roughly preserve the gradient magnitude.

We repeat the same numerical simulations as above for the STAR cell, and find that it indeed maintains healthy gradient magnitudes throughout most of the deep RNN, see Fig. 2 .

In the next section, we show also on real datasets that deep RNNs built from STAR units can be trained to a significantly greater depth.

As a final remark, the proposed modifications mean that the STAR architecture requires significantly less memory.

With the same input and the same capacity in the hidden state, it reduces the memory footprint to <40% of a classical LSTM and even uses slightly less memory than a recurrent highway net.

A more detailed comparison is given in the appendix.

We evaluate the performance of several well-known RNN baselines as well as that of the proposed STAR cell on five different sequence modelling tasks with three different datasets: sequential versions of MNIST, which are a popular common testbed for recurrent networks; the more realistic TUM dataset, where time series of intensities observed in satellite images shall be classified into different agricultural crops; and Jester, for hand gesture recognition with convolutional RNNs.

The recurrent units we compare include the vRNN, the LSTM, the LSTM with only a forget gate (van der Westhuizen & Lasenby, 2018) , the RHN, and the proposed STAR.

The experimental protocol is similar for all tasks: For each RNN variant, we train multiple versions with different depth (number of layers).

For each variant and each depth, we report the performance of the model with the lowest validation loss.

Classification performance is measured by the rate of correct predictions (top-1 accuracy).

Throughout, we use the orthogonal initialisation for weight matrices.

Code and trained models (in Tensorflow), as well as code for the simulations (in PyTorch), will be released.

Training and network details for each experiment can be found in the appendix.

97.0% 82.0% 100 11k uRNN (Arjovsky et al., 2016) 95.1% 91.4% 512 9k FC uRNN (Wisdom et al., 2016) 96.9% 94.1% 512 270k Soft ortho (Vorontsov et al., 2017) 94.1% 91.4% 128 18k AntisymRNN (Chang et al., 2019) 98

The first experiment uses the MNIST dataset (LeCun et al., 1998) .

The 28×28 grey-scale images of handwritten digits are flattened into 784×1 vectors, and the 784 values are sequentially presented to the RNN.

After seeing all pixels, the model predicts the digit.

The second task, pMNIST, is more challenging.

Before flattening the images, the pixels are shuffled with a fixed random permutation, turning correlations between spatially close pixels into non-local long-range dependencies.

The model needs to remember those dependencies between distance parts of the sequence to classify the digit correctly.

Fig. 3a shows the average gradient norms per layer at the start of training, for 12-layer networks built from different RNN cells.

Like in the simulations above, the propagation through the network increases the gradients for the vRNN and shrinks them for the LSTM.

As the optimisation proceeds, we find that STAR remains stable, whereas all other units see a rapid decline of the gradients already within the first epoch, except for RHN, where the gradients explode, see Fig. 3b .

Consequently, STAR is the only unit for which a 12-layer model can be trained, as also confirmed by the evolution of the training loss, Fig. 3c .

Fig. 4 confirms that stacking into deeper architectures does benefit RNNs (except for vRNN); but it increases the risk of a catastrophic training failure.

STAR is significantly more robust in that respect and can be trained up to a depth of 20 layers.

On the comparatively easy and saturated MNIST data, the performance is comparable that of a successfully trained LSTM (at depth 2-8 layers, LSTM training already often fails; the displayed accuracies are averaged only over successful training runs).

In this experiment, the models are evaluated on a more realistic sequence modelling problem.

The task is to classify agricultural crop types using sequences of satellite images, exploiting the fact that different crops have different growing patterns over the season.

The input is a time series of 26 multi-spectral Sentinel-2A satellite images with a ground resolution of 10 m, collected over a 102 km x 42 km area north of Munich, Germany between December 2015 and August 2016 (Rußwurm & Körner, 2017) .

The input data points for the classifier are patches of 3×3 pixels recorded in 6 spectral channels, flattened into 54×1 vectors.

In the first task these vectors are sequentially presented to the RNN model, which outputs a prediction at every time step (note that for this task the correct answer can sometimes be "cloud", "snow", "cloud shadow" or "water", which are easier to recognise than many crops).

In the second task, the model makes only one crop prediction for the complete sequence, via an additional layer that averages across time.

From Fig. 5 we see that STAR outperforms all baselines and its again more robust to stacking.

For the single-output task also the STAR network fails at 14 layers.

We have not yet been able to identify the reason for this, possibly it is due to cloud cover that completely blanks out the signal over extended time windows and degrades the propagation.

This experiment serves to evaluate the performance of different recurrent cells, and in particular the proposed STAR cell, in a convolutional RNN (see appendix for details about convolutional STAR).

To that end, we use the 20BN-Jester dataset V1 (jes).

Jester is a large collection of densely-labeled short video clips, where each clip contains a predefined hand gesture performed by a worker in front of a laptop camera or webcam.

In total, the dataset includes 148'094 RGB video files of 27 types of gestures.

The task is to classify which gesture is seen in a video.

32 consecutive frames of size 112×112 pixels are sequentially presented to the convolutional RNN.

At the end, the model again predicts a gesture class via an averaging layer over all time steps.

The outcome for convolutional RNNs is coherent with the previous results, see Fig. 5c .

Going deeper improves the performance of all three tested convRNNs.

The improvement is strongest for convolutional STAR, and the best performance is reached at high depth (12 layers), where training the baselines mostly fails.

In summary, the results confirm both our intuition that depth is particularly useful for convolutional RNNs; and that STAR is more suitable for deeper architectures, where it achieves higher performance with better memory efficiency.

We note that the in the shallow 1-2 layer setting the conventional LSTM performs a bit better than the two others, likely due to its larger capacity.

We have investigated the problem of vanishing/exploding gradient in deep RNNs.

In a first step, we analyse how the derivatives of the non-linear activation functions rescale the gradients as they propagate through the temporally unrolled network.

From both, the theoretical analysis, and associated numerical simulations, we find that standard RNN cells do not preserve the gradient magnitudes during backpropagation, and therefore, as the depth of the network grows, the risk that the gradients vanish or explode increases.

In a second step, we have proposed a new RNN cell, termed the STAckable Recurrent unit, which better preserves gradients through deep architectures and facilitates their training.

An extensive evaluation on three popular datasets confirms that STAR units can be stacked into deeper architectures than other RNN cells.

We see two main directions for future work.

On the one hand, it would be worthwhile to develop a more formal and thorough mathematical analysis of the gradient flow, and perhaps even derive rigorous bounds for specific cell types, that could, in turn, inform the network design.

On the other hand, it appears promising to investigate whether the analysis of the gradient flows could serve as a basis for better initialisation schemes to compensate the systematic influences of the cells structure, e.g., gating functions, in the training of deep RNNs.

C TRAINING DETAILS C.1 PIXEL-BY-PIXEL MNIST Following Tallec & Ollivier, chrono initialisation is applied for the bias term of k, b k .

The basic idea is that k should not be too large; such that the memory h can be retained over longer time intervals.

The same initialisation is used for the input and forget bias of the LSTM and the RHN and for the forget bias of LSTMw/f.

For the final prediction, a feedforward layer with softmax activation converts the hidden state to a class label.

The numbers of hidden units in the RNN layers are set to 128.

All networks are trained for 100 epochs with batch size 100, using the Adam optimizer (Kingma & Ba, 2014) with learning rate 0.001, β 1 = 0.9 and β 2 = 0.999.

For both tasks we use the same training schedule.

Again a feedforward layer is appended to the RNN output to obtain a prediction.

The numbers of hidden units in the RNN layers is set to 128.

All networks are trained for 30 epochs with batch size 500, using Adam (Kingma & Ba, 2014) with learning rate 0.001 and β 1 = 0.9 and β 2 = 0.999.

Throughout, convolution kernels are of size 3×3.

Each convolutional RNN layer has 64 filters.

A shallow CNN is used to convert the hidden state to a label, with 4 layers that have filter depths 128, 128, 256 and 256, respectively.

All models are fitted with stochastic gradient descent (SGD) with momentum (β = 0.9).

The batch size is set to 8, the learning rate starts at 0.001 and decays polynomially to 0.000001 over a total of 30 epochs.

L2-regularisation with weight 0.00005 is applied to all parameters.

@highlight

We analyze the gradient propagation in deep RNNs and from our analysis, we propose a new multi-layer deep RNN.