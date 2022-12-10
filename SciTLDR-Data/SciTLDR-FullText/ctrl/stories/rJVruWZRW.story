We propose the dense RNN, which has the fully connections from each hidden state to multiple preceding hidden states of all layers directly.

As the density of the connection increases, the number of paths through which the gradient flows can be increased.

It increases the magnitude of gradients, which help to prevent the vanishing gradient problem in time.

Larger gradients, however, can also cause exploding gradient problem.

To complement the trade-off between two problems, we propose an attention gate, which controls the amounts of gradient flows.

We describe the relation between the attention gate and the gradient flows by approximation.

The experiment on the language modeling using Penn Treebank corpus shows dense connections with the attention gate improve the model’s performance.

In order to analyze sequential data, it is important to choose an appropriate model to represent the data.

Recurrent neural network (RNN), as one of the model capturing sequential data, has been applied to many problems such as natural language , machine translation BID0 , speech recognition BID6 .

There are two main research issues to improve the RNNs performance: 1) vanishing and exploding gradient problems and 2) regularization.

The vanishing and exploding gradient problems occur as the sequential data has long-term dependency BID10 BID18 .

One of the solutions is to add gate functions such as the long short-term memory (LSTM) and gated recurrent unit (GRU).

The LSTM has additional gate functions and memory cells BID11 .

The gate function can prevent the gradient from being vanished during back propagation through time.

Gated recurrent unit (GRU) has similar performance with less gate functions BID1 .The part of sequential data whose boundary to distinguish the consecutive other parts, has the hierarchical structures.

To handle the hierarchy, the model should capture the multiple timescales.

In hierarchical multiple recurrent neural network (HM-RNN, Chung et al. (2016) ), the boundary information is also learned by implementing three operations such as update, copy and flush operator.

In clockwork RNN BID14 , the hidden states are divided into multiple sub-modules, which act with different periods to capture multiple timescales.

As all previous states within the recurrent depth do not always affect the next state, memory-augmented neural network (MANN, BID7 ) uses the memory to remember previous states and retrieve some of previous states if necessary.

The basic way to handle multiple timescales along with preventing the vanishing gradient problem is to increases both of feedforward depth and recurrent depth to capture multiple timescales.

Feedforward depth is the longest path from the input layer to the output layer.

Recurrent depth is the longest path from arbitrary hidden state at time t to same hidden sate at time t + t .

Increasing feedforward depth means stacking multiple recurrent layers deeply.

It can capture fast and slow changing components in the sequential data BID19 BID3 BID9 .

The low level layer in the stacked RNN captures short-term dependency.

As the layer is higher, the aggregated information from lower layer is abstracted.

Thus, as the layer is higher, the capacity to model long-term dependency increases.

The number of nonlinearities in the stacked RNN, however, is proportional to the number of unfolded time steps regardless of the feedforward depth.

Thus, the simple RNN and stacked RNN act identically in terms of long run.

Increasing recurrent depth also increases the capability to capture long-term dependency in the data.

The hidden state in vanilla RNN has only connection to previous time step's hidden state in the same layer.

Adding the connections to multiple previous time steps hidden states can make the shortcut paths, which alleviates the vanishing problem.

Nonlinear autoregressive exogenous model (NARX) handles the vanishing gradient problem by adding direct connections from the distant past in the same layer BID15 .

Similarly, higher-order RNN (HO-RNN) has the direct connections to multiple previous states with gating to each time step BID20 .

Unlike other recurrent models that use one connection between two consecutive time steps, the recurrent highway network (RHN) adds multiple connections with sharing parameters between transitions in the same layer BID23 .The vanilla RNN has only one path connected with previous hidden states.

Thus, it is hard to apply standard dropout technique for regularization as the information is being diluted during training of long-term sequences.

By selecting the same dropout mask for feedforward, recurrent connections, respectively, the dropout can apply to the RNN, which is called a variational dropout BID4 .

This paper proposes a dense RNN that has both of feedforward and recurrent depths.

The stacked RNN increases the complexity by increasing feedforward depth.

NARX-RNN and HO-RNN increase the complexity by increasing recurrent depth.

The model with the feedforward depth can be combined with the model with the recurrent depth, as the feedforward depth and recurrent depth have an orthogonal relationship.

Gated feedback RNN has the fully connection between two consecutive timesteps.

As the connection of gated feedback is not overlapped with the model with orthogonal depths, all three features, adding feedforward depth, recurrent depth, and gated feedback, can be modeled jointly .

With the three features, we propose the attention gate, which controls the flows from each state so that it enhances the overall performance.

The contributions of this paper are summarized: 1) dense RNN that is aggregated model with feedforward depth, recurrent depth and gated feedback function, 2) extension of the variational dropout to the dense RNN.

There are largely two methods to improve the performance of RNN.

One is to extend previous model by stacking multiple layers or adding gate functions.

The other is using regularization such as dropout to avoid overfitting.

In simple recurrent layer, h t , the hidden state at time t, is a function of input x t and preceding hidden state h t−1 , which is defined as follows: DISPLAYFORM0 where U and W are respectively the feedforward and recurrent weight matrix and φ means an element-wise nonlinear function such as T anh.

In simple recurrent layer, the last hidden state h t−1 has to memorize all historical inputs.

As the memorizing capacity of the hidden state is limited, it is hard to capture long-term dependency in sequential data.

Stacked recurrent neural network, stacked of the simple recurrent layers can capture the long-dependency, which is defined as follows: DISPLAYFORM1 where W j is the weight matrix for transition from layer j − 1 to j and U j is the weight matrix for transition from in timestep t − 1 to timestep t at layer j.

The stacked RNN can model multiple timescales of the sequential data.

As the information travels toward upper layer, the hidden state can memorize abstracted information of sequential data that covers more long-term dependency.

The other way to capture the long term dependency is to increase the recurrent depth by connecting the hidden state at timestep t directly to multiple preceding hidden states BID20 , Under review as a conference paper at ICLR 2018 which is defined as follows: DISPLAYFORM2 where U (k,j)→j is the weight matrix from layer j at timestep t − k to layer j at timestep t, and K is the recurrent depth.

The direct connections make the shortcut paths from preceding multiple hidden states.

Compared with the model without shortcut paths, the model with shortcut paths enables to access preceding hidden states further way from h j t with same number of transitions.

Most recurrent models have only recurrent connections between hidden states with same layers.

By adding feedback connections to the hidden state h j t from the preceding hidden states h i t−1 at different layer of h j t , the model adaptively captures the multiple timescales from long-term dependency, which is defined as follows: DISPLAYFORM3 where U i→j is the weight matrix from layer i at timestep t − 1 to layer j at timestep t, and L is the feedforward depth.

To control the amount of flows between different hidden states with different time scales BID1 , the global gate is applied as follows: DISPLAYFORM4 In FORMULA4 , g i→j is the gate to control the information flows from each hidden state to multiple preceding hidden states of all layers, which is defined as follows: DISPLAYFORM5 where w j g is a vector whose dimension is same with h DISPLAYFORM6 is a vector whose dimension is same with h * t−1 that is a concatenated vector of all hidden states from previous time step, and σ is an element-wise sigmoid function.

Gated feedback function is also applied to LSTM.

In gated LSTM, input gate i j t , forget gate f j t , output gate o j t , and memory cell gatec j t are defined as follows: DISPLAYFORM7 DISPLAYFORM8 DISPLAYFORM9 DISPLAYFORM10 Compared with conventional LSTM, the gated feedback LSTM has gated feedback function in the memory cell gatec j

As dropout is one method of neural network regularization, it prevents the model from being overfitted to training set.

However, it is hard to apply the standard dropout to recurrent connections.

As the sequence is longer, the information is affected by the dropout many time during backpropagation through time, which makes memorizing long sequences hard.

Thus, applying the standard dropout only to feedforward connections is recommended BID21 .

It is expressed as follows: DISPLAYFORM0 where D j t , as a dropout operator for every time step, makes h j−1 t being masked with Bernoulli dropout mask m j−1 t randomly generated for every time step.

BID17 proposed how to apply the dropout to recurrent connections efficiently.

By considering the whole sequential data as one input at a sequential level, same dropout mask is applied to recurrent connections at all time steps during training, which is expressed as follows: DISPLAYFORM1 where D j→j , as a time-independent dropout operator, makes h j t−1 being masked with Bernoulli dropout mask m j randomly generated regardless of time.

Gal & Ghahramani (2016) applied variational dropout to RNN and proved the relation between Bayesian inference and dropout theoretically.

Variational dropout in RNN applies same masks regardless of time to feedforward connections, similar to recurrent connections, which is expressed as follows: DISPLAYFORM2 3 DENSE RECURRENT NEURAL NETWORKThe skip connections that bypass some layers enables deep networks to be trained better than the models without skip connections.

Many research BID8 BID12 uses skip connections toward feedback connections.

In this paper, we apply the skip connections toward recurrent connections.

The shortcut paths from preceding multiple hidden states to the hidden state at time t is equal to the skip connections through time.

The shortcut paths include the feedback connections between different layers of different timesteps.

Each connection is controlled by the attention gate, similar to the global gate in the gated feedback RNN.

The dense RNN is defined as follows: DISPLAYFORM3 where g (k,i)→j is the attention gate, which is defined as follows: DISPLAYFORM4 (13) is a function of preceding hidden state at layer i and time t − k, while (6) is a function of concatenated all preceding hidden states.

The intuition why we change the gate function and we call it to attention gate is described in Section 3.1.The dense RNN can be easily extended to dense LSTM.

The dense LSTM is defined as follows: DISPLAYFORM5 DISPLAYFORM6 DISPLAYFORM7 DISPLAYFORM8 DISPLAYFORM9 Unlike gated feedback LSTM, the attention gate g is applied to all gates and memory cell state.

We analyze the advantages of dense connections intuitively and theoretically.

In addition, we propose the dropout method for dense connections.

Recurrent connections enable to predict next data given previous sequential data.

In the language modeling, the RNNs can predict next word based on the last word and the last context accumulated before the last word.

It assumes only last word affect to predict next word.

For instance, "the sky is" is given from the full sentence "the sky is blue" and the goal is to predict the word "blue".

In this case, the preceding word "sky provides the better clue than the preceding word "is".

Inspired by the fact, we propose the dense model that predicts the next word by directly referring to recent preceding words.

In other words, the output h j t is a function of input h j−1 t and recent preceding output h j t−k as in (3).

The higher the layer in a neural network, the more abstract the hidden states.

In language modeling, hidden states represent the characteristics of words, sentences, and paragraphs as the layer increases.

The conventional RNN has only the connection between same layer.

It means the preceding words, sentences, paragraphs determine next words, sentences and paragraphs, respectively.

The given word, however, can also determine the context of next paragraph.

Also, the given paragraph can determine next word.

For instance, the word "mystery" in "it is mystery" can be followed by the paragraph related to "mystery" and vice versa.

The feedback connections can reflect the fact.

In FORMULA3 , preceding words, sentences, and paragraphs affects next words, sentences, and paragraphs with same scale.

Preceding words, however, dont affects next word prediction evenly.

In the sentence "The sky is blue", the word "sky" has a very close relation with the word "blue".

The word "The", as an article, has a less relation with the word "blue".

The amount how two words are related depends on the kind of the two words.

We define the the degree of relevance as gated attention g as in (5).The attention g is determined by the preceding word itself and the last word given as input.

In the sentence "The sky is blue", the features of the word "the", and "sky" denote h t−2 , and h t−1 , respectively and the word "is" denotes x t or h 0 t .

Then, the attention to predict the word"blue" from the word "The" is determined by the word "The and "blue".

The attention to predict the word "blue" from the word "is" is determined by the word "is" and "blue".

In other words, the attention is dependent on the previous hidden state h j t−k and input h j−1 t at certain time step as in (13).

The vanishing and exploding gradient problems happen the sequential data has long term dependency.

During backpropagation through time, error E T 's gradient with respect to the recurrent weight matrix U j is vanished or exploded as the sequence is longer, which is expressed as follows: DISPLAYFORM0 The critical term related to vanishing and exploding gradient problems is ∂h j τ /∂h j τ −1 .

To find the relation between vanishing and exploding gradient problems and dense connections, we assume the worst situation in which the vanishing and exploding gradient problem may arise.

The term ∂h j τ /∂h j τ −1 denotes A j .

If the A j max is less than 1, the gradient with respect to U j would be exploded and if the A j min is greater than 1, gradient with respect to U j would be vanished.

In dense recurrent network, there are more paths to flow the gradients.

The A j in dense recurrent network is approximated as follows: DISPLAYFORM1 where the superscript (k, i) → j means the direction of the path from h DISPLAYFORM2 , which reduces the vanishing gradient boundary from 1 to 1/(KL) as shown in FIG1 (a).Though dense connections are able to alleviate the vanishing gradient problem, it causes the gradient exploding.

To alleviate the problem caused by dense connection, we add the attention gates as in (13).

The attention gate g (k,i)→j can control the magnitude of A (k,i)→j , which is expressed as follows: DISPLAYFORM3 In FORMULA0 , the g (k,i)→j is trainable so that the vanishing and exploding boundary is determined adaptively.

In dense RNN with attention gates, h i τ +k−1 is expressed as follows: DISPLAYFORM4 where θ is not relevant parameters with h DISPLAYFORM5 .

For simplicity, FORMULA0 is expressed as y = φ(g(U g x) · U x + θ).

Gradient of y with respect to x is calculated as follows: DISPLAYFORM6 The FORMULA0 is scaled withg compared to ∂y ∂x = y(1 − y)U without attention gate g. As g andg are similar as shown in FIG1 (c),g is approximated g as in (19).In recurrent highway network (RHN, Zilly et al. (2016) ), the effect of highway was described using the Geršgorin circle theorem (GST, Geršhgorin (1931) ).

Likewise, the dffect of the attention gate in the proposed model can be interpreted using GST.

For simplicity, we only formulate recurrent connection with omitting feedforward connection, h t+1 = φ(U h t ).

Then, the Jacobian matrix DISPLAYFORM7 .

By letting γ be a maximal bound on diag[φ (U h t )] and ρ max be the largest singular value of U T , the norm of the Jacobian satisfies using the triangle inequality as follows: DISPLAYFORM8 The value γρ max is less than 1, the vanishing gradient problem happens and the A is greater than 1, the exploding gradient problem happens as the range of the A has no explicit boundary.

The spectrum of A, the set of λ in A, is evaluated as follows: DISPLAYFORM9 which means the eigenvalue λ lies on the circle whose radius is the summation of abstract values of all elements except the diagonal element a ii and center is the diagonal element a ii .

The simplified recurrent connection with the attention gate is h t+1 = φ(gU h t ) where g is U g , h t .

Then, the Jacobian matrix A = ∂ht+1 ∂ht is expressed as follows: DISPLAYFORM10 The spectrum of FORMULA1 is expressed as follows: DISPLAYFORM11 The scaled term term g + g U g,i h t,i in (23) can be approximated as g as shown in FIG1 .

Thus, the upper bound of A is approximately less than 1 so that the attention gate g can alleviate the exploding problem.

In dense RNN, as recurrent depth increases, the number of parameters also increases, which makes the model vulnerable to overfitting to training dataset.

To prevent the model from overfitting to training dataset, the dropout is applied.

The variational dropout, proved to show good performance in the RNN models, uses same random masks at every time step for feedforward connections, and recurrent connections, respectively.

In implementation of variational dropout, each state is dropped with the random mask, which is followed by weighted sum.

This concept is extensively applied to dense RNN so that the same random mask is used regardless of time and recurrent depth.

Extension of variational dropout to the dense connection is expressed as follows: DISPLAYFORM12 4 EXPERIMENTIn our experiment, we used Penn Tree corpus (PTB) for language modeling.

PTB consists of 923k training set, 73k validation set, and 82k test set.

As a baseline model, we select the model proposed by BID21 , which proposed how to predict next word based on the previous words.

To improve the accuracy, BID21 proposed regularization method for RNN.The baseline models hidden sizes are fixed as 200, 650, and 1500, which are called as small, medium, and large network.

In this paper, we fixed hidden size as 200 for all hidden layers with varying the feedforward depth from 2 to 3, recurrent depth from 1 to 5.

The word embedding weights were tied with word prediction weights.

To speed up the proposed method, all of matrix multiplications in (12) was performed by the batch matrix-matrix product of two matrices.

Each of batches is rewritten as follows: DISPLAYFORM13 The FORMULA0 is rewritten as follows: DISPLAYFORM14 The proposed dense model is trained with stochastic gradient decent.

The initial rate was set to 20, which was decayed with 1.1 after the epoch was over 12.

The training was terminated when the epoch reaches 120.

To prevent the exploding gradient, we clipped the max value of gradient as 5.As a regularization, we adopt variational dropout, which uses the random masks regardless of time.

We configured the word embeddings dropout to 0.3, feedforward connections dropout 0.2, and recurrent connections dropout rate varies from 0.2 to 0.5.

The TAB0 , as a trained result, compares the baseline model, RNN model with variational dropout and using same word space.

In dense RNN, the perplexity is better than two models.

The best models recurrent depth is 2 and the perplexity of valid set is 83.28 and that of test set is 78.82.

This paper proposed dense RNN, which has fully connections from each hidden state to multiple preceding hidden states of all layers directly.

Each previous hidden state has its attention gate that controls the amount of information flows.

To evaluate the effect of dense connections, we used Penn Treebank corpus (PTB).

The result of dense connection was confirmed by varying the recurrent depth with the attention gate.

The dense connections with the attention gate made the model's perplexity less than conventional RNN.

<|TLDR|>

@highlight

Dense RNN that has fully connections from each hidden state to multiple preceding hidden states of all layers directly.

@highlight

Proposes a new RNN architecture that models long-term dependencies better, can learn multiscale representation of sequential data, and sidestep the gradients problem by using parametrized gating units.

@highlight

This paper proposes a fully connected dense RNN architecture with gated connections to every layer and preceding layer connections, and it's results on PTB charcter-level modelling task.