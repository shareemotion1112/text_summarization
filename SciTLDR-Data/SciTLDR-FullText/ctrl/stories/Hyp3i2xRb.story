Plain recurrent networks greatly suffer from the vanishing gradient problem while Gated Neural Networks (GNNs) such as Long-short Term Memory (LSTM) and Gated Recurrent Unit (GRU) deliver promising results in many sequence learning tasks through sophisticated network designs.

This paper shows how we can address this problem in a plain recurrent network by analyzing the gating mechanisms in GNNs.

We propose a novel network called the Recurrent Identity Network (RIN) which allows a plain recurrent network to overcome the vanishing gradient problem while training very deep models without the use of gates.

We compare this model with IRNNs and LSTMs on multiple sequence modeling benchmarks.

The RINs demonstrate competitive performance and converge faster in all tasks.

Notably, small RIN models produce 12%–67% higher accuracy on the Sequential and Permuted MNIST datasets and reach state-of-the-art performance on the bAbI question answering dataset.

Numerous methods have been proposed for mitigating the vanishing gradient problem including the use of second-order optimization methods (e.g., Hessian-free optimization BID15 ), specific training schedules (e.g., Greedy Layer-wise training BID20 BID7 BID24 ), and special weight initialization methods when training on both plain FFNs and RNNs BID3 BID16 BID13 BID10 BID26 BID11 .Gated Neural Networks (GNNs) also help to mitigate this problem by introducing "gates" to control information flow through the network over layers or sequences.

Notable examples include recurrent networks such as Long-short Term Memory (LSTM) BID8 , Gated Recurrent Unit (GRU) BID1 , and feedforward networks such as Highway Networks (HNs) BID21 , and Residual Networks (ResNets) BID5 .

One can successfully train very deep models by employing these models, e.g., ResNets can be trained with over 1,000 layers.

It has been demonstrated that removing (lesioning) or reordering (re-shuffling) random layers in deep feedforward GNNs does not noticeable affect the performance of the network BID23 Noticeably, one interpretation for this effect as given by BID4 is that the functional blocks in HNs or ResNets engage in an Unrolled Iterative Estimate (UIE) of representations and that layers in this block of HNs or ResNets iteratively refine a single set of representations.

In this paper, we investigate if the view of Iterative Estimation (IE) can also be applied towards recurrent GNNs (Section 2.1).

We present a formal analysis for GNNs by examining a dual gate design common in LSTM and GRU (Section 2.2).

The analysis suggests that the use of gates in GNNs encourages the network to learn an identity mapping which can be beneficial in training deep architectures BID6 BID4 .We propose a new formulation of a plain RNN, called a Recurrent Identity Network (RIN) , that is encouraged to learn an identity mapping without the use of gates (Section 2).

This network uses ReLU as the activation function and contains a set of non-trainable parameters.

This simple yet effective method helps the plain recurrent network to overcome the vanishing gradient problem while it is still able to model long-range dependencies.

This network is compared against two competing networks, the IRNN (Le et al., 2015) and LSTM, on several long sequence modeling tasks including the adding problem (Section 3.1), Sequential and Permuted MNIST classification tasks (Section 3.2), and bAbI question answering tasks (Section 3.3).

RINs show faster convergence than IRNNs and LSTMs in the early stage of the training phase and reach competitive performance in all benchmarks.

Note that the use of ReLU in RNNs usually leads to training instability, and therefore the network is sensitive to training hyperparameters.

Our proposed RIN network demonstrates that a plain RNN does not suffer from this problem even with the use of ReLUs as shown in Section 3.

We discuss further implications of this network and related work in Section 4.

Representation learning in RNNs requires that the network build a latent state, which reflects the temporal dependencies over a sequence of inputs.

In this section, we explore an interpretation of this process using iterative estimation (IE), a view that is similar to the UIE view for feedforward GNNs.

Formally, we characterize this viewpoint in Eq. 1, that is, the expectation of the difference between the hidden activation at step t, h t , and the last hidden activation at step T , h T , is zero: DISPLAYFORM0 This formulation implies that an RNN layer maintains and updates the same set of representations over the input sequence.

Given the fact that the hidden activation at every step is an estimation of the final activation, we derive Eq. 3.

Average Estimation Error DISPLAYFORM1 DISPLAYFORM2 Figure 1: Observation of learning identity mapping in an LSTM model trained on the adding problem task (see Section 3.1).

The average estimation error is computed over a batch of 128 samples of the test set.

(a) and (b) show the evaluation of Eq. 1 and Eq. 3 respectively.

The x-axis indicates the index of the step that compares with the final output h T or its previous step h t−1 .

Fig. 1 shows an empirical observation of the IE in the adding problem (experimental details in Section 3.1).

Here, we use the Average Estimation Error (AEE) measure BID4 to quantify the expectation of the difference between two hidden activations.

The measured AEEs in Fig. 1 are close to 0 indicating that the LSTM model fulfills the view of IE.

The results also suggest that the network learns an identity mapping since the activation levels are similar on average across all recurrent updates.

In the next section, we shall show that the use of gates in GNNs encourages the network to learn an identity mapping and whether this analysis can be extended to plain recurrent networks.

Popular GNNs such as LSTM, GRU; and recent variants such as the Phased-LSTM BID17 , and Intersection RNN BID2 , share the same dual gate design following: DISPLAYFORM0 where t ∈ [1, T ], H t = σ(x t , h t−1 ) represents the hidden transformation, T t = τ (x t , h t−1 ) is the transform gate, and C t = φ(x t , h t−1 ) is the carry gate.

σ, τ and φ are recurrent layers that have their trainable parameters and activation functions.

represents element-wise product operator.

Note that h t may not be the output activation at the recurrent step t. For example in LSTM, h t represents the memory cell state.

Typically, the elements of transform gate T t,k and carry gate C t,k are between 0 (close) and 1 (open), the value indicates the openness of the gate at the kth neuron.

Hence, a plain recurrent network is a subcase of Eq. 4 when T t = 1 and C t = 0.Note that conventionally, the initial hidden activation h 0 is 0 to represent a "void state" at the start of computation.

For h 0 to fit into Eq. 4's framework, we define an auxiliary state h −1 as the previous state of h 0 , and T 0 = 1, C 0 = 0.

We also define another auxiliary state h T +1 = h T , T T +1 = 0, and C T +1 = 1 as the succeeding state of h T .Based on the recursive definition in Eq. 4, we can write the final layer output h T as follows: DISPLAYFORM1 where we use to represent element-wise multiplication over a series of terms.

According to Eq. 3, and supposing that Eq. 5 fulfills the Eq. 1, we can use a zero-mean residual t for describing the difference between the outputs of recurrent steps: DISPLAYFORM2 Plugging Eq. 6 into Eq. 5, we get DISPLAYFORM3 The complete deduction of Eqs. 8-9 is presented in Appendix A. Eq. 8 performs an identity mapping when the carry gate C t is always open.

In Eq. 9, the term t i=1 i represents "a level of representation that is formed between h 1 and h t ".

Moreover, the term T j=t C j extract the "useful" part of this representation and contribute to the final representation of the recurrent layer.

Here, we interpret "useful" as any quantity that helps in minimizing the cost function.

Therefore, the contribution, λ t , at each recurrent step, quantifies the representation that is learned in the step t. Furthermore, it is generally believed that a GNN manages and maintains the latent state through the carry gate, such as the forget gate in LSTM.

If the carry gate is closed, then it is impossible for the old state to be preserved while undergoing recurrent updates.

However, if we set C t = 0, t ∈ [1, T ] in Eq. 9, we get: DISPLAYFORM4 If h 0 = 0 (void state at the start), we can turn Eq. 10 into: DISPLAYFORM5 Eq. 11 shows that the state can be preserved without the help of the carry gate.

This result indicates that it is possible for a plain recurrent network to learn an identity mapping as well.

Motivated by the previous iterative estimation interpretation of RNNs, we formulate a novel plain recurrent network variant -Recurrent Identity Network (RIN): DISPLAYFORM0 where W is the input-to-hidden weight matrix, U is the hidden-to-hidden weight matrix, and I is a non-trainable identity matrix that acts as a "surrogate memory" component.

This formulation encourages the network to preserve a copy of the last state by embedding I into the hidden-tohidden weights.

This "surrogate memory" component maintains the representation encoded in the past recurrent steps.

In this section, we compare the performances of the RIN, IRNN, and LSTM in a set of tasks that require modeling long-range dependencies.

The adding problem is a standard task for examining the capability of RNNs for modeling longrange dependencies BID8 .

In this task, two numbers are randomly selected from a long sequence.

The network has to predict the sum of these two numbers.

The task becomes challenging as the length of the sequence T increases because the relevant numbers can be far from each other in a long sequence.

We report experimental results from three datasets that have sequence lengths of T 1 = 200, T 2 = 300, and T 3 = 400 respectively.

Each dataset has 100,000 training samples and 10,000 testing samples.

Each sequence of a dataset has T i numbers that are randomly sampled from a uniform distribution in [0, 1].

Each sequence is accompanied by a mask that indicates the two chosen random positions.

We compare the performance between RINs, IRNNs, and LSTMs using the same experimental settings.

Each network has one hidden layer with 100 hidden units.

Note that a LSTM has four times more parameters than corresponding RIN and IRNN models.

The optimizer minimizes the Mean Squared Error (MSE) between the target sum and the predicted sum.

We initially used the RMSprop BID22 optimizer.

However, some IRNN models failed to converge using this optimizer.

Therefore, we chose the Adam optimizer (Kingma & Ba, 2014) so a fair comparison can be made between the different networks.

The batch size is 32.

Gradient clipping value for all models is 100.

The models are trained with maximum 300 epochs until they converged.

The initial learning rates are different between the datasets because we found that IRNNs are sensitive to the initial learning rate as the sequence length increases.

The learning rates α 200 = 10 −4 , α 300 = 10 DISPLAYFORM0 and α 400 = 10 −6 are applied to T 1 , T 2 and T 3 correspondingly.

The input-to-hidden weights of RINs and IRNNs and hidden-to-hidden weights of RINs are initialized using a similar method to BID13 where the weights are drawn from a Gaussian distribution N (0, 10 −3 ).

The LSTM is initialized with the settings where the input-to-hidden weights use Glorot Uniform BID3 and hidden-to-hidden weights use an orthogonal matrix as suggested by BID19 .

Bias values for all networks are initialized to 0.

No explicit regularization is employed.

We do not perform an exhaustive hyperparameter search in these experiments.

The baseline MSE of the task is 0.167.

This score is achieved by predicting the sum of two numbers as 1 regardless of the input sequence.

FIG1 shows MSE plots for different test datasets.

RINs and IRNNs reached the same level of performance in all experiments, and LSTMs performed the worst.

Notably, LSTM fails to converge in the dataset with T 3 = 400.

The use of ReLU in RINs and IRNNs causes some degree of instability in the training phase.

However, in most cases, RINs converge faster and are more stable than IRNNs (see training loss plots in Fig. 5 of Appendix B).

Note that because IRNNs are sensitive to the initial learning rate, applying high learning rates such as α = 10 −3 for T 2 and T 3 could cause the training of the network to fail.

Sequential and Permuted MNIST are introduced by Le et al. FORMULA0 for evaluating RNNs.

Sequential MNIST presents each pixel of the MNIST handwritten image BID14 to the network sequentially (e.g., from the top left corner of the image to the bottom right corner of the image).

After the network has seen all 28 × 28 = 784 pixels, the network produces the class of the image.

This task requires the network to model a very long sequence that has 784 steps.

Permuted MNIST is an even harder task than the Sequential MNIST in that a fixed random index permutation is applied to all images.

This random permutation breaks the association between adjacent pixels.

The network is expected to find the hidden relations between pixels so that it can correctly classify the image.

All networks are trained with the RMSprop optimizer BID22 ) and a batch size of 128.

The networks are trained with maximum 500 epochs until they are converged.

The initial learning rate is set to α = 10 −6 .

Weight initialization follows the same setup as Section 3.1.

No explicit regularization is added.

TAB0 summarizes the accuracy performance of the networks on the Sequential and Permuted MNIST datasets.

For small network sizes (1-100, 1-200), RINs outperform IRNNs in their accuracy performance.

For bigger networks, RINs and IRNNs achieve similar performance; however, RINs converge much faster than IRNNs in the early stage of training (see FIG2 ).

LSTMs perform the worst on both tasks in terms of both convergence speed and final accuracy.

Appendix C presents the full experimental results.

To investigate the limit of RINs, we adopted the concept of Deep Transition (DT) Networks BID18 for increasing the implicit network depth.

In this extended RIN model called RIN-DT, each recurrent step performs two hidden transitions instead of one (the formulation is given in Appendix D).

The network modification increases the inherent depth by a factor of two.

The results showed that the error signal could survive 784 × 2 = 1568 computation steps in RIN-DTs.

In FIG3 , we show the evidence of learning identity mapping empirically by collecting the hidden activation from all recurrent steps and evaluating Eqs. 1 and 3.

The network matches the IE when AEE is close to zero.

We also compute the variance of the difference between two recurrent steps.

FIG3 suggests that all networks bound the variance across recurrent steps.

FIG3 (b) offers a closer perspective where it measures the AEE between two adjacent steps.

The levels of activations for all networks are always kept the same on an average, which is an evidence of learning identity mapping.

We also observed that the magnitude of the variance becomes significantly larger at the last 200 steps in IRNN and RIN.

Repeated application of ReLU may cause this effect during recurrent update BID9 .

Other experiments in this section exhibit similar behaviors, complete results are shown in Appendix C FIG1 ).

Note that this empirical analysis only demonstrates that the tested RNNs have the evidence of learning identity mapping across recurrent updates as RINs and IRNNs largely fulfill the view of IE.

We do not over-explain the relationship between this analysis and the performance of the network. .

The x-axis indicates the index of the step that compares with the final output h T or its previous step h t−1 , and y-axis represents the average estimation error (AEE).

DISPLAYFORM0

The bAbI dataset provides 20 question answering tasks that measure the understanding of language and the performance of reasoning in neural networks BID25 .

Each task consists of 1,000 training samples and 1,000 test samples.

A sample consists of three parts: a list of statements, a question and an answer (examples in TAB1 ).

The answer to the question can be inferred from the statements that are logically organized together.

The red square is below the blue square.

Then she journeyed to the garden.

The red square is to the left of the pink rectangle.

Question: Is the blue square below the pink rectangle?

Answer: Garden.

Answer:

No.

We compare the performance of the RIN, IRNN, and LSTM on these tasks.

All networks follow a network design where the network firstly embeds each word into a vector of 200 dimensions.

The statements are then appended together to a single sequence and encoded by a recurrent layer while another recurrent layer encodes the question sequence.

The outputs of these two recurrent layers are concatenated together, and this concatenated sequence is then passed to a different recurrent layer for decoding the answer.

Finally, the network predicts the answer via a softmax layer.

The recurrent layers in all networks have 100 hidden units.

This network design roughly follows the architecture presented in BID11 .

The initial learning rates are set to α = 10 −3 for RINs and LSTMs and α = 10 −4 for IRNNs because IRNNs fail to converge with a higher learning rate on many tasks.

We chose the Adam optimizer over the RMSprop optimizer because of the same reasons as in the adding problem.

The batch size is 32.

Each network is trained for maximum 100 epochs until the network converges.

The recurrent layers in the network follow the same initialization steps as in Section 3.1.The results in TAB2 show that RINs can reach mean performance similar to the state-of-theart performance reported in BID11 .

As discussed in Section 3.1, the use of ReLU as the activation function can lead to instability during training of IRNN for tasks that have lengthy statements (e.g.. 3-Three Supporting Facts, 5-Three Arg.

Relations).

In this paper, we discussed the iterative representation refinement in RNNs and how this viewpoint could help in learning identity mapping.

Under this observation, we demonstrated that the contribution of each recurrent step a GNN can be jointly determined by the representation that is formed up to the current step, and the openness of the carry gate in later recurrent updates.

Note in Eq. 9, the element-wise multiplication of C t s selects the encoded representation that could arrive at the output of the layer.

Thus, it is possible to embed a special function in C t s so that they are sensitive to certain pattern of interests.

For example, in Phased LSTM, the time gate is inherently interested in temporal frequency selection BID17 .Motivated by the analysis presented in Section 2, we propose a novel plain recurrent network variant, the Recurrent Identity Network (RIN), that can model long-range dependencies without the use of gates.

Compared to the conventional formulation of plain RNNs, the formulation of RINs only adds a set of non-trainable weights to represent a "surrogate memory" component so that the learned representation can be maintained across two recurrent steps.

Experimental results in Section 3 show that RINs are competitive against other network models such as IRNNs and LSTMs.

Particularly, small RINs produce 12%-67% higher accuracy in the Sequential and Permuted MNIST.

Furthermore, RINs demonstrated much faster convergence speed in early phase of training, which is a desirable advantage for platforms with limited computing resources.

RINs work well without advanced methods of weight initializations and are relatively insensitive to hyperparameters such as learning rate, batch size, and selection of optimizer.

This property can be very helpful when the time available for choosing hyperparameters is limited.

Note that we do not claim that RINs outperform LSTMs in general because LSTMs may achieve comparable performance with finely-tuned hyperparameters.

The use of ReLU in RNNs might be counterintuitive at first sight because the repeated application of this activation is more likely causing gradient explosion than conventional choices of activation function, such as hyperbolic tangent (tanh) function or sigmoid function.

Although the proposed IRNN BID13 reduces the problem by the identity initialization, in our experiments, we usually found that IRNN is more sensitive to training parameters and more unstable than RINs and LSTMs.

On the contrary, feedforward models that use ReLU usually produce better results and converge faster than FFNs that use the tanh or sigmoid activation function.

In this paper, we provide a promising method of using ReLU in RNNs so that the network is less sensitive to the training conditions.

The experimental results also support the argument that the use of ReLU significantly speeds up the convergence.

During the development of this paper, a recent independent work BID27 presented a similar network formulation with a focus on training of deep plain FFNs without skip connections.

DiracNet uses the idea of ResNets where it assumes that the identity initialization can replace the role of the skip-connection in ResNets.

DiracNet employed a particular kind of activation function -negative concatenated ReLU (NCReLU), and this activation function allows the layer output to approximate the layer input when the expectation of the weights are close to zero.

In this paper, we showed that an RNN can be trained without the use of gates or special activation functions, which complements the findings and provides theoretical basis in BID27 .We hope to see more empirical and theoretical insights that explains the effectiveness of the RIN by simply embedding a non-trainable identity matrix.

In future, we will investigate the reasons for the faster convergence speed of the RIN during training.

Furthermore, we will investigate why RIN can be trained stably with the repeated application of ReLU and why it is less sensitive to training parameters than the two other models.

A ALGEBRA OF EQS.

8-9Popular GNNs such as LSTM, GRU; and recent variants such as the Phased-LSTM BID17 , and Intersection RNN BID2 , share the same dual gate design described as follows: DISPLAYFORM0 where t ∈ [1, T ], H t = σ(x t , h t−1 ) represents the hidden transformation, T t = τ (x t , h t−1 ) is the transform gate, and C t = φ(x t , h t−1 ) is the carry gate.

σ, τ and φ are recurrent layers that have their trainable parameters and activation functions.

represents element-wise product operator.

Note that h t may not be the output activation at the recurrent step t. For example in LSTM, h t represents the memory cell state.

Typically, the elements of transform gate T t,k and carry gate C t,k are between 0 (close) and 1 (open), the value indicates the openness of the gate at the kth neuron.

Hence, a plain recurrent network is a subcase of Eq. 14 when T t = 1 and C t = 0.Note that conventionally, the initial hidden activation h 0 is 0 to represent a "void state" at the start of computation.

For h 0 to fit into Eq. 4's framework, we define an auxiliary state h −1 as the previous state of h 0 , and T 0 = 1, C 0 = 0.

We also define another auxiliary state h T +1 = h T , T T +1 = 0, and C T +1 = 1 as the succeeding state of h T .Based on the recursive definition in Eq. 4, we can write the final layer output h T as follows: DISPLAYFORM1 where we use to represent element-wise multiplication over a series of terms.

According to Eq. 3, and supposing that Eq. 5 fulfills the Eq. 1, we can use a zero-mean residual t for describing the difference between the outputs of recurrent steps: DISPLAYFORM2 Then we can rewrite Eq. 16 as: DISPLAYFORM3 Substituting Eq. 18 into Eq. 15: DISPLAYFORM4 We can rearrange Eqn.

20 to DISPLAYFORM5 The term λ in Eq. 23 can be reorganized to, DISPLAYFORM6 B DETAILS IN THE ADDING PROBLEM EXPERIMENTS Average Estimation Error RIN 2-100 1st IRNN 2-100 1st LSTM 2-100 1st 0 100 200 300 400 500 600 700 800 layer 2 step index RIN 2-100 2nd IRNN 2-100 2nd LSTM 2-100 2nd DISPLAYFORM7 DISPLAYFORM8

In Section 3.2, we tested an additional model for RINs, which takes the concept of Deep Transition Networks (DTNs) BID18 .

Instead of stacking the recurrent layers, DTNs add multiple nonlinear transitions in a single recurrent step.

This modification massively increases the depth of the network.

In our RIN-DTs, the number of transition per recurrent step is two.

Because the length of the sequence for Sequential and Permuted MNIST tasks is 784, RIN-DTs have the depth of 784 × 2 = 1568.

The recurrent layer is defined in Eqs. 26-27.

DISPLAYFORM0 DISPLAYFORM1

<|TLDR|>

@highlight

We propose a novel network called the Recurrent Identity Network (RIN) which allows a plain recurrent network to overcome the vanishing gradient problem while training very deep models without the use of gates.