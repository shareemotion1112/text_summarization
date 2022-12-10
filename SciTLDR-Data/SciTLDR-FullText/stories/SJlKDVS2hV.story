Recurrent  neural  networks  (RNNs)  are  a  powerful tool for modeling sequential data.

Despite their widespread usage, understanding how RNNs solve complex problems remains elusive.

Here, we characterize how popular RNN architectures perform document-level sentiment classification.

Despite their theoretical capacity to implement complex, high-dimensional computations, we find that  trained  networks  converge  to  highly  interpretable, low-dimensional representations.

We identify a simple mechanism, integration along an approximate line attractor, and find this mechanism present across RNN architectures (including LSTMs, GRUs, and vanilla RNNs).

Overall, these results demonstrate that surprisingly universal and human interpretable computations can arise across a range of recurrent networks.

Recurrent neural networks (RNNs) are a popular tool for sequence modelling tasks.

These architectures are thought to learn complex relationships in input sequences, and exploit this structure in a nonlinear fashion.

RNNs are typically viewed as black boxes, despite considerable interest in better understanding how they function.

Here, we focus on studying how recurrent networks solve document-level sentiment analysis-a simple, but longstanding benchmark task for language modeling BID6 BID13 .

We demonstrate that popular RNN architectures, despite having the capacity to implement high-dimensional and nonlinear computations, in practice converge to low-dimensional representations when trained against this task.

Moreover, using analysis techniques from dynamical systems theory, we show that locally linear approximations to the nonlinear RNN dynamics are highly interpretable.

In particular, they all involve approximate low-dimensional line attractor dynamics-a useful dynamical feature that can be implemented by linear dynamics and used to store an analog value BID10 .

Furthermore, we show that this mechanism is surprisingly consistent across a range of RNN architectures.

We trained four RNN architectures-LSTM BID3 , GRU BID0 , Update Gate RNN (UGRNN) BID1 , and standard (vanilla) RNNs-on binary sentiment classifcation tasks.

We trained each network type on each of three datasets: the IMDB movie review dataset, which contains 50,000 highly polarized reviews BID7 ; the Yelp review dataset, which contained 500,000 user reviews BID14 ; and the Stanford Sentiment Treebank, which contains 11,855 sentences taken from movie reviews BID11 .

For each task and architecture, we analyzed the best performing networks, selected using a validation set (see Appendix A for details).

We analyzed trained networks by linearizing the dynamics around approximate fixed points.

Approximate fixed points are state vectors {h 1 N h − F (h, 0) 2 2 , and then minimizing q with respect to hidden states, h, using standard auto-differentiation methods BID2 .

We ran this optimization multiple times starting from different initial values of h. These initial conditions were sampled randomly from the state activation visited by the trained network, which was done to intentionally sample states related to the operation of the RNN.

For brevity, we explain our approach using the working example of the LSTM trained on the Yelp dataset FIG0 ).

We find similar results for all architectures and datasets, these are shown in Appendix B.

As an initial exploratory analysis step, we performed principal components analysis (PCA) on the RNN states concatenated across 1,000 test examples.

The top 2-3 PCs explained ∼90% of the variance in hidden state activity FIG0 , black line).

The distribution of hidden states visited by untrained networks on the same test set was much higher dimensional FIG0 , gray line), suggesting that training the networks stretched the geometry of their representations along a lowdimensional subspace.

We then visualized the RNN dynamics in this lowdimensional space by forming a 2D histogram of the density of RNN states colored by the sentiment label FIG0 .

This visualization is reminiscent of integration dynamics along a line attractor-a well-studied mechanism for evidence accumulation in simple recurrent networks BID10 BID8 )-and we reasoned that similar dynamics may be used for sentiment classification.

The hypothesis that RNNs approximate line attractor dynamics during sentiment classification makes four specific predictions, which we investigate and confirm in subsequent sections.

First, the fixed points form an approximately 1D manifold that is aligned/correlated with the readout weights (Section 3.2).

Second, all fixed points are attracting and marginally stable.

That is, in the absence of input (or, perhaps, if a string of neutral/uninformative words are encountered) the RNN state should rapidly converge to the closest fixed point and then should not change appreciably (Section 3.4).

Third, locally around each fixed point, inputs representing positive vs. negative evidence should produce linearly separable effects on the RNN state vector along some dimension (Section 3.5).

Finally, these instantaneous effects should be integrated by the recurrent dynamics along the direction of the 1D fixed point manifold (Section 3.5).

We numerically identified the location of ∼500 RNN fixed points using previously established methods BID12 BID2 .

We then projected these fixed points into the same low-dimensional space used in FIG0 .

Although the PCA projection was fit to the RNN hidden states, and not the fixed points, a very high percentage of variance in fixed points was captured by this projection FIG0 , suggesting that that the RNN states remain close to the manifold of fixed points.

We call the vector that describes the main axis of variation of the 1D manifold m. Consistent with the line attractor hypothesis, the fixed points appeared to be spread along a 1D curve when visualized in PC space, and furthermore the principal direction of this curve was aligned with the readout weights FIG0 .

We further verified that this low-dimensional approximation was accurate by using locally linear embedding (LLE; BID9 to parameterize a 1D manifold of fixed points in the raw, high-dimensional data.

This provided a scalar coordinate, θ i ∈ [−1, 1], for each fixed point, which was well-matched to the position of the fixed point manifold in PC space (coloring of points in FIG0 ).

We next aimed to demonstrate that the identified fixed points were marginally stable, and thus could be used to preserve accumulated information from the inputs.

To do this, we used a standard linearization procedure BID4 ) to obtain an approximate, but highly interpretable, description of the RNN dynamics near the fixed point manifold.

Given the last state h t−1 and the current input x t , the approach is to locally approximate the update rule with a first-order Taylor expansion: DISPLAYFORM0 where ∆h t−1 = h t−1 − h * and ∆x t = x t − x * , and {J rec , J inp } are Jacobian matrices of the system: DISPLAYFORM1 We choose h * to be a numerically identified fixed point and x * = 0, thus we have F (h * , x * ) ≈ h * and ∆x t = x t .

Under this choice, equation FORMULA0 reduces to a discrete-time linear dynamical system: DISPLAYFORM2 It is important to note that both Jacobians depend on which fixed point we choose to linearize around, and should thus be thought of as functions of h * ; for notational simplicity we do not denote this dependence explicitly.

By reducing the nonlinear RNN to a linear system, we can analytically estimate the network's response to a sequence of T inputs.

In this approximation, the effect of each input x t is decoupled from all others; that is, the final state is given by the sum of all individual effects.

1 We can restrict our focus to the effect of a single input, x t (i.e. a single term in this sum).

Let k = T − t be the number of time steps between x t and the end of the document.

The total effect of x t on the final RNN state becomes: DISPLAYFORM3 1 We consider the case where the network has closely converged to a fixed point, so that h0 = h * and thus ∆h0 = 0.where L = R −1 , the columns of R (denoted r a ) contain the right eigenvectors of J rec , the rows of L (denoted a ) contain the left eigenvectors of J rec , and Λ is a diagonal matrix containing complex-valued eigenvalues, λ 1 > λ 2 > . . .

> λ N , which are sorted based on their magnitude.

From equation 3 we see that x t affects the representation of the network through N terms (called the eigenmodes or modes of the system).

The magnitude of each mode after k steps is given by the λ k a ; thus, the size of each mode either reduces to zero or diverges exponentially fast, with a time constant given by: τ a = 1 log(|λa|) .

This time constant has units of tokens (or, roughly, words) and yields an interpretable number for the effective memory of the system.

FIG1 plots the eigenvalues and associated time constants and shows the distribution of all eigenvalues at three representative fixed points along the fixed point manifold FIG1 .

In FIG1 , we plot the decay time constant of the top three modes; the slowest decaying mode persists after ∼1000 time steps, while the next two modes persists after ∼100 time steps, with lower modes decaying even faster.

Since the average review length for the Yelp dataset is ∼175 words, only a small number of modes could represent information from the beginning of the document.

Overall, these eigenvalue spectra are consistent with our observation that RNN states only explore a low-dimensional subspace when performing sentiment classification.

RNN activity along the majority of dimensions is associated with fast time constants and is therefore quickly forgotten.

Restricting our focus to the top eigenmode for simplicity (there may be a few slow modes of integration), the effect of a single input, x t , on the network activity (equation 3) becomes: r 1 1 J inp x, where we have dropped the dependence on t since λ 1 ≈ 1, so the effect of x is largely insensitive to the exact time it was input to system.

Using this expression, we separately analyzed the effects of specific words.

We first examined the term J inp x for various choices of x (i.e. various word tokens).

This quantity represents the instantaneous linear effect of x on the RNN state vector and is shared across all eigenmodes.

We projected the resulting vectors onto the same low-dimensional subspace shown in FIG0 .

We see that positive and negative valence words push the hidden state in opposite directions.

Neutral words, in contrast, exert much smaller effects on the RNN state FIG2 .While J inp x represents the instantaneous effect of a word, only the features of this input that overlap with the top few eigenmodes are reliably remembered by the network.

The scalar quantity 1 J inp x, which we call the input projection, captures the magnitude of change induced by x along the eigenmode associated with the longest timescale.

Again we observe that the valence of x strongly correlates with this quantity: neutral words have an input projection near zero while positive and negative words produced larger magnitude responses of opposite sign.

Furthermore, this is reliably observed across all fixed points.

FIG2 shows the average input projection for positive, negative, and neutral words; the histogram shows the distribution of these average effects across all fixed points along the line attractor.

Finally, if the input projection onto the top eigenmode is non-negligible, then the right eigenvector r 1 (which is normalized to unit length) represents the direction along which x is integrated.

If the RNN implements an approximate line attractor, then r 1 (and potentially other slow modes) should align with the principal direction of the manifold of fixed points, m. We indeed observe a high degree of overlap between r 1 and m both visually in PC space FIG2 and quantitatively across all fixed points FIG2 .

In this work we applied dynamical systems analysis to understand how RNNs solve sentiment analysis.

We found a simple mechanismintegration along a line attractorpresent in multiple architectures trained to solve the task.

Overall, this work provides preliminary, but optimistic, evidence that different, highly intricate network models can converge to similar solutions that may be reduced and understood by human practitioners.

An RNN is defined by a nonlinear update rule h t = F (h t−1 , x t ), which is applied recursively from an initial state h 0 over a sequence of inputs x 1 , x 2 , . . .

, x T .

Let N and M denote the dimensionality of the hidden states and the input vectors, so that h t ∈ R N and x t ∈ R M .

In sentiment classification, T represents the number of word tokens in a sequence, which can vary on a document-by-document basis.

To process word sequences for a given dataset, we build a vocabulary and encode words as one-hot vectors.

These are fed to a dense linear embedding layer with an embedding size of M = 128 (x t are the embeddings in what follows).

The word embeddings were trained from scratch simultaneously with the RNN.

We considered four RNN architectures-LSTM BID3 , GRU BID0 , UGRNN BID1 , and the vanilla RNN (VRNN)-each corresponding to a separate nonlinear update rule, F (·, ·).

For the LSTM architecture, h t consists of a concatenated hidden state vector and cell state vector so that N is twice the number of computational units; in all other architectures N is equal to the number of units.

The RNN prediction is evaluated at the final time step T , and is given byŷ = w h T + b, where we call w ∈ R N the readout weights.

In the LSTM architecture, the cell state vector is not read out, and thus half of the entries in w are enforced to be zero under this notation.

We examined three benchmark datasets for sentiment classification: the IMDB movie review dataset, which contains 50,000 highly polarized reviews BID7 ; the Yelp review dataset, which contained 500,000 user reviews BID14 ; and the Stanford Sentiment Treebank, which contains 11,855 sentences taken from movie reviews BID11 .

The Stanford Sentiment Treebank also contains short phrases with labeled sentiments; these were not analyzed.

For each task and architecture, we performed a randomized hyper-parameter search and selected the best networks based on a validation set.

All models were trained using Adam BID5 ) with a batch size of 64.

The hyper-parameter search was performed over the following ranges: the initial learning rate (10 −5 to 10 −1 ), learning rate decay factor (0 to 1), gradient norm clipping (10 −1 to 10), 2 regularization penalty (10 −3 to 10 −1 ), the β 1 (0.5 to 0.99), and β 2 (0.9999 to 0.99) parameters of the Adam optimization routine.

We additionally trained a bag-of-words model (logistic regression trained with word counts), as a baseline comparison.

The accuracies of our final models on the held-out test set are summarized in Table 1 .

We analyzed the best performing models for each combination of architecture type and dataset.

For each model, we numerically identified a large set of fixed points {h * BID12 .

Briefly, we accomplished this by first defining the loss function q = 1 N h−F (h, 0) 2 2 , and then minimizing q with respect to hidden states, h, using standard auto-differentiation methods BID2 .

We ran this optimization multiple times starting from different initial values of h. These initial conditions were sampled randomly from the state activation during the operation of the trained network, which was done to intentionally sample states related to the operation of the RNN.

We varied the stopping tolerance for q using 9 points logarithmically spaced between 10 −9 and 10 −5 running the optimization from 1000 different initial conditions for each tolerance.

This allowed us to find approximate fixed points of varying speeds.

Values of q at numerical zero are true fixed points, while small but non-zero values are called slow points.

Slow points are often reasonable places to perform a linearization, assuming that √ q, which is akin to speed, is slow compared to the operation of the network.

DISPLAYFORM0

Below we provide figures summarizing the linear integration mechanism for each combination of architecture (LSTMs, GRUs, Update Gate RNNs, Vanilla RNNs) and dataset (Yelp, IMDB, and Stanford Sentiment).

Note that the first figure, LSTMs trained on Yelp, reproduces the figures in the main text-we include it here for completeness.

The description of each panel is given in FIG0 , note that these descriptions are the same across all figures.

We find that these mechanisms are remarkably consistent across architectures and datasets. (lower left) Instantaneous effect of word inputs, J inp x, for positive (green), negative (red), and neutral (cyan) words.

Blue arrows denote 1, the top left eigenvector.

The PCA projection is the same as FIG1 , but centered around each fixed point.(lower middle left) Average of 1 J inp x over 100 different words, shown for positive, negative, neutral words.(lower middle right) Same plot as in FIG1 , with an example fixed point highlighted (approximate fixed points in grey).

Blue arrows denote r1, the top right eigenvector.(lower right) Distribution of r 1 m (overlap of the top right eigenvector with the fixed point manifold) over all fixed points.

Null distribution is randomly generated unit vectors of the size of the hidden state.

@highlight

We analyze recurrent networks trained on sentiment classification, and find that they all exhibit approximate line attractor dynamics when solving this task.