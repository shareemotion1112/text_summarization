In this paper, we present a layer-wise learning of stochastic neural networks (SNNs) in an information-theoretic perspective.

In each layer of an SNN, the compression and the relevance are defined to quantify the amount of information that the layer contains about the input space and the target space, respectively.

We jointly optimize the compression and the relevance of all parameters in an SNN to better exploit the neural network's representation.

Previously, the Information Bottleneck (IB) framework (\cite{Tishby99}) extracts relevant information for a target variable.

Here, we propose Parametric Information Bottleneck (PIB) for a neural network by utilizing (only) its model parameters explicitly to approximate the compression and the relevance.

We show that, as compared to the maximum likelihood estimate (MLE) principle, PIBs : (i) improve the generalization of neural networks in classification tasks, (ii) push the representation of neural networks closer to the optimal information-theoretical representation in a faster manner.

Deep neural networks (DNNs) have demonstrated competitive performance in several learning tasks including image recognition (e.g., BID14 , ), natural language translation (e.g., , ) and game playing (e.g., BID22 ).

Specifically in supervised learning contexts, a common practice to achieve good performance is to train DNNs with the maximum likelihood estimate (MLE) principle along with various techniques such as data-specific design of network architecture (e.g., convolutional neural network architecture), regularizations (e.g., early stopping, weight decay, dropout BID25 ), and batch normalization BID12 )), and optimizations (e.g., BID13 ).

The learning principle in DNNs has therefore attributed to the MLE principle as a standard one for guiding the learning toward a beneficial direction.

However, the MLE principle is very generic that is not specially tailored for neural networks.

Thus, a reasonable question is does the MLE principle effectively and sufficiently exploit a neural network's representative power and is there any better alternative?

As an attempt to address this important question, this work investigates the learning of DNNs from the information-theoretic perspective.

An alternative principle is the Information Bottleneck (IB) framework BID29 ) which extracts relevant information in an input variable X about a target variable Y .

More specifically, the IB framework constructs a bottleneck variable Z = Z(X)

that is compressed version of X but preserves as much relevant information in X about Y as possible.

In this information-theoretic perspective, I(Z, X) 1 , the mutual information of Z and X, captures the compression of Z about X and I(Z, Y ) represents the relevance of Z to Y .

The optimal representation Z is determined via the minimization of the following Lagrangian: DISPLAYFORM0 where β is the positive Lagrangian multiplier that controls the trade-off between the complexity of the representation, I(Z, X), and the amount of relevant information in Z, I(Z, Y ).

The exact solution to the minimization problem above is found BID29 ) with the implicit selfconsistent equations: DISPLAYFORM1 p(z) = p(z|x)p(x)dx p(y|z) = p(y|x)p(x|z)dx (2) where Z(x; β) is the normalization function, and D KL [. .] is the Kullback -Leibler (KL) divergence BID15 ).

Unfortunately, the self-consistent equations are highly non-linear and still non-analytic for most practical cases of interest.

Furthermore, the general IB framework assumes that the joint distribution p(X, Y ) is known and does not specify concrete models.

On the other hand, the goal of the MLE principle is to match the model distribution p model as close to the empirical data distributionp D as possible (e.g., see Appendix I.B).

The MLE principle treats the neural network model p(x x x; θ θ θ) as a whole without explicitly considering the contribution of its internal structures (e.g., hidden layers and hidden neurons).

As a result, a neural network with redundant information in hidden layers may have a good distribution match in a training set but show a poor generalization in test sets.

In the MLE principle, we only need empirical samples of the joint distribution to maximize the likelihood function of the model given the data.

The MLE principle is proved to be mathematically equivalent to the IB principle for the multinomial mixture model for clustering problem when the input distribution X is uniform or has a large sample size BID24 ).

However in general the two principles are not obviously related.

In this work, we leverage neural networks and the IB principle by viewing neural networks as a set of encoders that sequentially modify the original data space.

We then propose a new generalized IB-based objective that takes into account the compression and relevance of all layers in the network as an explicit goal for guiding the encodings in a beneficial manner.

Since the objective is designed to optimize all parameters of neural networks and is mainly motivated by the IB principle for deep learning BID28 ), we name this method the Parametric Information Bottleneck (PIB).

Because the generalized IB objective in PIB is intractable, we approximate it using variational methods and Monte Carlo estimation.

We propose re-using the existing neural network architecture as variational decoders for each hidden layers.

The approximate generalized IB objective in turn presents interesting connections with the MLE principle.

We show that our PIBs have a better generalization and better exploit the neural network's representation by pushing it closer to the information-theoretical optimal representation as compared to the MLE principle.

Originally, the general IB framework is proposed in BID29 .

The framework provides a principled way of extracting the relevant information in one variable X about another variable Y .

The authors represent the exact solution to the IB problem in highly-nonlinear self-consistent equations and propose the iterative Blahut Arimoto algorithm to optimize the objective.

However, the algorithm is not applicable to neural networks.

In practice, the IB problem can be solved efficiently in the following two cases only: (1) X, Y and Z are all discrete BID29 ); or (2) X, Y and Z are mutually joint Gaussian BID6 ) where Z is a bottleneck variable.

Recently, the IB principle has been applied to DNNs BID28 ).

This work proposes using mutual information of a hidden layer with the input layer and the output layer to quantify the performance of DNNs.

By analyzing these measures with the IB principle, the authors establish an information-theoretic learning principle for DNNs.

In theory, one can optimize the neural network by pushing up the network and all its hidden layers to the IB optimal limit in a layerwise manner.

Although the analysis offers a new perspective about optimality in neural networks, it proposes general analysis of optimality rather than a practical optimization criteria.

Furthermore, estimating mutual information between the variables transformed by network layers and the data variables poses several computational challenges in practice that the authors did not address in the work.

A small change in a multi-layered neural network could greatly modify the entropy of the input variables.

Thus, it is hard to analytically capture such modifications.

The recent work BID2 also uses variational methods to approximate the mutual information as an attempt to apply the IB principle to neural networks.

Their approach however considers one single bottleneck and parameterizes the encoder p(z z z|x x x; θ θ θ) by an entire neural network.

The encoder maps the input variable x x x to a single bottleneck variable z z z that is not a part of the considered neural network architecture.

Therefore, their approach still treats a neural network as a whole rather than optimizing it layer-wise.

Furthermore, the work imposes a variational prior distribution in the code space to approximate its actual marginal distribution.

However, the variational approximate distribution for the code space may be too loose while the actual marginal distribution can be sampled easily.

Our work, on the other hand, focuses on better exploiting intermediate representations of a neural network architecture using the IB principle.

More specifically, our work proposes an optimization IB criteria for an existing neural network architecture in an effort to better learn the layers' representation to their IB optimality.

In estimating mutual information, we adopted the variational method as in BID2 for I(Z, Y ) but use empirical estimation for I(Z, X).

Furthermore, we exploit the existing network architecture as variational decoders rather than resort to variational decoders that are not part of the neural network architecture.

This section presents an information-theoretic perspective of neural networks and then defines our PIB framework.

This perspective paves a way for the soundness of constraining the compressionrelevance trade-off into a neural network.

We denote X, Y as the input and the target (label) variables of the data, respectively; Z l as a stochastic variable represented by the l th hidden layer of a neural network where 1 ≤ l ≤ L, L is the number of hidden layers.

We extend the notations of Z l by using the convention Z 0 := X and Z −1 := ∅. The space of X, Y and Z l are denoted as X , Y and Z l , respectively.

Each respective space is associated with the corresponding probability measures p D (x x x), p D (y y y) and p(z z z l ) where p D (.) indicates the underlying probability distribution of the data and p(.) denotes model distributions.

Each Z l is stochastically mapped from the previous stochastic variable Z l−1 via an encoder p(z z z l |z z z l−1 ).

We name Z l , 1 ≤ l ≤ L as a (information) bottleneck or code variable of the network.

In this work, we focus on binary bottlenecks where Z l ∈ {0, 1} n l and n i is the dimensionality of the bottleneck space.

An encoder p(z z z|x x x) introduces a soft partitioning of the space X into a new space Z whose probability measure is determined as p(z z z) = p(z z z|x x x)p D (x x x)dx x x. The encoding can modify the information content of the original space possibly including its dimensionality and topological structure.

On average, 2 H(X|Z) elements of X are mapped to the same code in Z. Thus, the average volume of a partitioning of X is 2 H(X) /2 H(X|Z) = 2 I(X,Z) .

The mutual information I(Z, X) which measures the amount of information that Z contains about X can therefore quantify the quality of the encoding p(z z z|x x x).

A smaller mutual information I(Z, X) implies a more compressed representation Z in terms of X.Since the original data space is continuous, it requires infinite precision to represent it precisely.

However, only some set of underlying explanatory factors in the the data space would be beneficial for a certain task.

Therefore, lossy representation is often more helpful (and of course more efficient) than a precise representation.

In this aspect, we view the hidden layers of a multi-layered neural network as a lossy representation of the data space.

The neural network in this perspective consists of a series of stochastic encoders that sequentially encode the original data space X into the intermediate code spaces Z l .

These code spaces are lossy representations of the data space as it follows from the data-processing inequality (DPI) BID8 ) that DISPLAYFORM0 where we assume that Y, X, Z l and Z l+1 form a Markov chain in that order, i.e., DISPLAYFORM1 Figure 1: A directed graphical representation of a PIB of two bottlenecks.

The neural network parameters θ θ θ = (θ θ θ 1 , θ θ θ 2 , θ θ θ 3 ).

The dashed blue arrows do not denote variable dependencies but the relevance decoders for each bottleneck.

The relevance decoder p true (y y y|z z z i ), which is uniquely determined given the encoder p θ θ θ (z z z i |x x x) and the joint distribution p D (x x x, y y y), is intractable.

We use p θ θ θ (y y y|z z z i ) as a variational approximation to each intractable relevance decoder p true (y y y|z z z i ).A learning principle should compress irrelevant information and preserve relevant information in the lossy intermediate code spaces.

In the next subsection, we describe in details how a sequential series of encoders, compression and relevance are defined in a neural network.

Our PIB framework is an extension of the IB framework to optimize all paramters of neural networks.

In neural networks, intermediate representations represent a hierarchy of information bottlenecks that sequentially extract relevant information for a target from the input data space.

Existing IB framework for DNNs specifies a single bottleneck while our PIB preserves hierarchical representations which a neural network's expressiveness comes from.

Our PIB also gives neural networks an information-theoretic interpretation both in network structure and model learning.

In PIBs, we utilize only neural network parameters θ θ θ for defining encoders and variational relevance decoders at every level, therefore the name Parametric Information Bottleneck.

Our PIB is also a standard step towards better exploiting representational power of more expressive neural network models such as Convolutional Neural Networks ) and ResNet BID11 ).

In this paper, we focus on binary bottlenecks in which the encoder p(z z z l |z z z l−1 ) is defined as DISPLAYFORM0 where DISPLAYFORM1 σ(.) is the sigmoid function, and W (l) is the weights connecting the l th layer to the (l + 1) th layer.

Depending on the structure of the target space Y, we can use an appropriate model for output distributions as follows: FORMULA0 For classification, we model the output distribution with softmax function, DISPLAYFORM2 The conditional distribution p(y y y|x x x) from the model is computed using the Bayes' rule and the Markov assumption (Equation 4) in PIBs 2 : DISPLAYFORM3 where z z z = (z z z 1 , z z z 2 , ..., z z z L ) is the entire sequence of hidden layers in the neural network.

Note that for a given joint distribution p D (x x x, y y y), the relevance decoder p true (y y y|z z z l ) is uniquely determined if an encoding function p(z z z l |x x x) is defined.

Specifically, the relevance decoder is determined as follows: DISPLAYFORM4 It is also important to note that many stochastic neural networks have been proposed before (e.g., BID18 , BID19 , BID27 , BID21 , BID9 ).

However, our motivation for this stochasticity is that it enables bottleneck sampling given the data variables (X, Y ).

The generated bottleneck samples are then used to estimate mutual information.

Thus, our framework does not depend on a specific stochastic model.

For deterministic neural networks, we only have one sample of hidden variables given one data point.

Thus, estimating mutual information for hidden variables in this case is as hard as estimating mutual information for the data variables themselves.

Since the neural network is a lossy representation of the original data space, a learning principle should make this loss in a beneficial manner.

Specifically in PIBs, we propose to jointly compress the network's intermediate spaces and preserve relevant information simultaneously at all layers of the network.

For the l th -level bottleneck Z l , the compression is defined as the mutual information between Z l and the previous-level bottleneck Z l−1 while the relevance is specified as its mutual information with the target variable Y .

We explicitly define the learning objective for PIB as: DISPLAYFORM0 where the layer-specific Lagrangian multiplier β −1 l controls the tradeoff between relevance and compression in each bottleneck, and the concept of compression and relevance is taken to the extreme when l = 0 (with convention that I(Z 0 , Z −1 ) = I(X, ∅) = H(X) = constant).

Here we prefer to this extreme, i.e., the 0 th level, as the super level.

While the l th level for 1 ≤ l ≤ L indicates a specific hidden layer l, the super level represents the entire neural network as a whole.

The objective L P IB can be considered as a joint version of the theoretical IB analysis for DNNs in BID28 .

However, minimizing L P IB has an intuitive interpretation as tightening the "information knots" of a neural network architecture simultaneously at every layer level (including the super level).

Optimizing PIBs now becomes the minimization of L P IB (Z) which attempts to decrease I(Z l , Z l−1 ) and increase I(Z l , Y ) simultaneously.

The decrease of I(Z l , Z l−1 ) makes the representation at the l th -level more compressed while the increase of I(Z l , Y ) promotes the preservation of relevant information in Z l about Y .

In optimization's aspect, the minimization of L P IB is much harder than the minimization of L IB since L P IB involves inter-dependent terms that even the self-consistent equations of the IB framework are not applicable to this case.

Furthermore, L P IB is intractable since the bottleneck spaces are usually high-dimensional and the relevance encoders p true (y y y|z z z l ) (computed by Equation 8) are intractable.

In the following section, we present our approximation to L P IB which fully utilizes the existing architecture without resorting to any model that is not part of the considered neural network.

The approximation then leads to effective gradient-based training of PIBs.

Here, we present our approximations to the relevance and the compression terms in the PIB objective L P IB .

Since the relevance decoder p true (y y y|z z z l ) (Equation 8) is intractable, we use a variational relevance decoder p v (y y y|z z z l ) to approximate it.

Firstly, we decompose the mutual information into a difference of two entropies: DISPLAYFORM0 where H(Y ) = constant can be ignored in the minimization of L(Z), and H(Y |Z l ) = − p true (y y y|z z z l )p(z z z l ) log p true (y y y|z z z l )dy y ydz z z l= − p D (x x x, y y y)p(z z z l |x x x) log p true (y y y|z z z l )dz z z l dx x xdy y y= − p D (x x x, y y y)p(z z z l |x x x) log p v (y y y|z z z l )dz z z l dx x xdy y y DISPLAYFORM1 ≤ − p D (x x x, y y y)p(z z z l |x x x) log p v (y y y|z z z l )dz z z l dx x xdy y y (14) DISPLAYFORM2 where the equality in Equation 12 holds due to the Markov assumption (Equation 4).

In PIBs, we propose to use the higher-level part of the existing network architecture at each layer to define the variational relevance encoder for that layer, i.e., p v (y y y|z z z l ) = p(y y y|z z z l ) where p(y y y|z z z l ) is determined by the network architecture.

In this case, we have: DISPLAYFORM3 We will refer toH(Y |Z l ) as the variational conditional relevance (VCR) for the l th -level bottleneck variable Z l for the rest of this work.

In the following, we present two important results which indicate that the relevance terms in our objective is closely and mutually related to the concept of the MLE principle.

Proposition 3.1.

The VCR at the super level (i.e., l = 0) equals the negative log-likelihood (NLL) function.

Proposition 3.2.

The VCR at the highest-level bottleneck variable Z L equals the VCR for the entire compositional bottleneck variable Z = (Z 1 , Z 2 , ..., Z L ) which is an upper bound on the NLL.

That DISPLAYFORM4 While the Proposition 3.1 is a direct result of Equation 16, the Proposition 3.2 holds due to Jensen's inequality (its detail derivation in Appendix I.A).In PIB's terms, the MLE principle can be interpreted as increasing the VCR of the network as a whole while the PIB objective takes into account the VCR at every level of the network.

In turn, the VCR can also be interpreted in terms of the MLE principle as follows.

It follows from Equation 15 and 16 that the VCR for layer l (including l = 0) is the NLL function of p(y y y|z z z l ).

Therefore, increasing the Relevance parts of J P IB is equivalent to performing the MLE principle for every layer level instead of the only super level as in the standard MLE.

Another interpretation is that our PIB framework encourages forwarding explicit information from all layer levels for better exploitation during learning while the MLE principle performs an implicit information forwarding by using only information from the super level.

Finally, the VCR for a multivariate y y y can be decomposed into the sum of that for each component of y y y (see Appendix I.C).

The compression terms in L P IB involve computing mutual information between two consecutive bottlenecks.

For simplicity, we present the derivation of I(Z 1 , Z 0 ) only 3 .

For the compression, we decompose the mutual information as follows: DISPLAYFORM0 which consists of the entropy and conditional entropy term.

The conditional entropy can be further rewritten as: DISPLAYFORM1 where DISPLAYFORM2 and H(Z 1,i |Z 0 = z z z 0 ) = −q log q − (1 − q) log(1 − q) where q = p(Z 1,i = 1|Z 0 = z z z 0 ).

For the entropy term H(Z 1 ), we resort to empirical samples of z z z 1 generated by Monte Carlo sampling to estimate the entropy: DISPLAYFORM3 where z z z DISPLAYFORM4 This estimator is also known as the maximum likelihood estimator or 'plug-in' estimator BID3 ).

The larger number of samples M guarantees the better plug-in entropy by the following bias bound BID20 ) DISPLAYFORM5 where |Z 1 | denotes the cardinality of the space of variable Z 1 .

In practice, log p(z z z 1 ) may be numerically unstable for large cardinality |Z 1 |.

In the large space of Z 1 , the probability of a single point p(z z z 1 ) may become very small that log p(z z z 1 ) becomes numerically unstable.

To overcome this problem, we propose an upper bound on the entropy using Jensen's inequality: DISPLAYFORM6 The upper boundH(Z 1 ) is numerically stable because the conditional distribution p(z z z 1 |z z z 0 ) is factorized into i p(z 1,i |z z z 0 ), therefore, log p(z z z 1 |z z z 0 ) = i log p(z 1,i |z z z 0 ) which is more stable.

The upper boundH(Z 1 ) can then be estimated using Monte Carlo sampling for z z z 0 and z z z 1 .

Discrete-valued variables in PIBs make standard back-propagation not straightforward.

Fortunately, one can estimate the gradient in this case.

The authors in BID27 used a Generalized EM algorithm while BID5 proposed to resort to reinforcement learning.

However, these estimators have high variance.

In this work, we use the gradient estimator inspired by BID21 for binary bottlenecks because it has low variance despite of being biased.

Specifically, a bottleneck z z z = (z 1 , z 2 , ..., z n l ) can be rewritten as being continuous by z i = σ(a i ) + i where i = 1 − σ(a i ) with probability σ(a i ) −σ(a i ) with probability 1 − σ(a i )The bottleneck component z i defined as above still gets value of either 0 or 1 but it is decomposed into the sum of a deterministic term and a noise term.

The gradient is then propagated only through the deterministic term and ignored in the noise term.

A detail of gradient-based training of PIB is presented in Algorithm 1.

One advantage of GRAD-P IB algorithm is that it requires only a single forward pass to estimate all the information terms inL P IB since the generated samples are re-used to compute the information terms at each layer level.

Use the generated samples above and Equations 15 and 23 to approximateL P IB (θ θ θ)

g g g ← ∂ ∂θ θ θL P IB (θ θ θ) using Raiko estimator 9:θ θ θ ← Update parameters using the approximate gradients g g g and SGD 10: until convergence of parameters θ θ θ 11: Output: θ θ θ 12: end procedure

We used the same architectures for PIBs and Stochastic Feed-forward Neural Networks (SFNNs) (e.g., BID27 ) and trained them on the MNIST dataset ) for image classification, odd-even decision problem and multi-modal learning.

Here, a SFNN simply prefers to feed-forward neural network models following the MLE principle for learning model parameters.

Each hidden layer in SFNNs is also considered as a stochastic variable.

The aforementioned tasks are to evaluate PIBs, as compared to SFNNs, in terms of generalization, learning dynamics, and capability of modeling complicated output structures, respectively.

All models are implemented using Theano framework (Al-Rfou et al. FORMULA0 ).

In this experiment, we compare PIBs with SFNNs and deterministic neural networks in the classification task.

For comparisons, we trained PIBs and five additional models.

The first model (Model A) is a deterministic neural network.

In Model D, we used the weight trained in Model A to perform stochastic prediction at test time.

Model E is SFNN and Model B is Model C with deterministic prediction during test phase.

Model C uses the weighted trained in PIB but we report deterministic prediction instead of stochastic prediction for test performance.

Mean (%) Std dev.

deterministic (A)1.73 -deterministic SFNN as deterministic (B)1.88 -PIB as deterministic (C) The MNIST dataset (LeCun (1998)) contains a standard split of 60000, and 10000 examples of handwritten digit images for training and test, respectively in which each image is grayscale of size 28 × 28 pixels.

We used the last 10000 images of the training set as a holdout set for tuning hyperparameters.

The best configuration chosen from the holdout set is used to retrain the models from scratch in the full training set.

The result in the test set is then reported (for stochastic prediction, we report mean and standard deviation).

We scaled the images to [0, 1] and do not perform any other data augmentation.

These base configurations are applied to all six models we use in this experiment.

The base architecture is a fully-connected, sigmoid activation neural network with two hidden layers and 512 units per layer.

Weights are initialized using Xavier initialization BID10 ).

Models were optimized with stochastic gradient descent with a constant learning rate of 0.1 and a batch size of 8.

For stochastic sampling, we generate M = 16 samples per point during training and M = 32 samples per point during testing.

For stochastic prediction, we run the prediction 10 times and report its mean and deviation standard.

For PIBs, we set β l = β, ∀1 ≤ l ≤ L.

We tuned β from {0} ∪ {10 −i : 1 ≤ i ≤ 7}, and found β −1 = 10 −4 works best.

Table 1 provides the results in the MNIST classification error in the test set for PIB and the comparative models (A), (B), (C), (D), and (E).

As can be seen from the table, PIB and Model C gives nearly the same performance which outperform deterministic neural networks and SFNNs, and their stochastic and deterministic version.

It is interesting to empirically see that the deterministic version of PIB at test time (Model C) gives a slightly better result than PIB.

This also empirically holds for the case of SFNN.

To investigate more in this, we compute the test error for various values of the number of samples used for Monte-Carlo averaging, M FIG1 ).

As we can see from the figure, the Monte-Carlo averaging of PIB obtains its good approximation around M = 30 and the deterministic prediction roughly places a lower bound on the Monte-Carlo averaging at test time.

For visualization of learned filters of PIB, see Appendix II.A.

One way to visualize the learning dynamic of each layer of a neural network is to plot the layers in the information plane BID29 , BID23 ).

The information plane is an informationtheoretic plane that characterizes any representation Z = Z(X) in terms of (I(Z, Y ), I(Z, X)) given the joint distribution I(X, Y ).

The plane has I(Z, X) and I(Z, Y ) as its horizontal axis and its vertical axis, respectively.

In the general IB framework, each value of β specifies a unique point of Z in the information plane.

As β varies from 0 to ∞, Z traces a concave curve, known as information curve for representation Z, with a slope of β −1 .

The information-theoretic goal of learning a representation Z = Z(X)

is therefore to push Z as closer to its corresponding optimal point in the information curve as possible.

For multi-layered neural networks, each hidden layer Z l is a representation that can also be quantified in the information plane.

In this experiment, we considered an odd-even decision problem in the MNIST dataset in which the task is to determine if the digit in an image is odd or even.

We used the same neural network architecture of 784-10-10-10-1 for PIB and SFNN and trained them with SGD with constant learning rate of 0.01 in the first 50000 training samples.

We used three different randomly initialized neural DISPLAYFORM0 networks and averaged the mutual informations.

For PIB, we used β DISPLAYFORM1 Since the network architecture is small, we can compute mutual information I x := I(Z i , X) and I y := I(Z i , Y ) precisely and plot them over training epochs.

As indicated by FIG2 , both PIB and SFNN enable the network to gradually encode more information into their hidden layers at the beginning as I(Z i , X) increases.

The encoded information at the beginning also contains some relevant information for the target variable as I(Z i , Y ) increases as well.

However, information encoding in the PIB is more selective as it quickly encodes more relevant information (it reaches higher I(Z, Y ) but in lesser number of epochs) while keeps the layers concise at higher epochs.

The SFNN, on the other hand, encodes information in a way that matches the model distribution to the empirical data distribution.

As a result, it may encode irrelevant information that hurts the generalization.

For additional visualization, an empirical architecture analysis of PIB and SFNN is presented in Appendix II.B.

As PIB and SFNN are stochastic neural networks, they can model structured output space in which a one-to-many mapping is required.

A binary stochastic variable z z z l of dimensionality n l can take on 2 n l different states each of which would give a different y y y.

This is the reason why the conditional distribution p(y y y|x x x) in stochastic neural networks is multi-modal.

In this experiment, we followed BID21 and predicted the lower half of the MNIST digits using the upper half as inputs.

We used the same neural network architecture of 392-512-512-392 for PIB and SFNN and trained them with SGD with constant learning rate of 0.01.

We trained the models in the full training set of 60000 images and tested in the test set.

For PIB, we also used β −1 l = β −1 = 10 −4 .

The visualization in Figure 4 indicates that PIB models the structured output space better and faster (using lesser number of epochs) than SFNN.

The samples generated by PIB is totally recognizable while the samples generated by SFNN shows some discontinuity (e.g., digit 2, 4, 5, 7) and confusion (e.g., digit 3 confuses with number 8, digit 5 is unrecognizable or confuses with number 6, digit 8 and 9 are unrecognizable).

In this paper we introduced an information-theoretic learning framework to better exploit a neural network's representation.

We have also proposed an approximation that fully utilizes all parameters in a neural network and does not resort to any extra models.

Our learning framework offers a principled way of interpreting and learning all layers of neural networks and encourages a more Figure 4 : Samples drawn from the prediction of the lower half of the MNIST test data digits based on the upper half for PIB (left, after 60 epochs) and SFNN (right, after 200 epochs).

The leftmost column is the original MNIST test digit followed by the masked out digits and nine samples.

The rightmost column is obtained by averaging over all generated samples of bottlenecks drawn from the prediction.

The figures illustrate the capability of modeling structured output space using PIB and SFNN.

informative yet compressed representation, which is supported by qualitative empirical results.

One limitation is that we consider here fully-connected feed-forward architecture with binary hidden layers.

Since we used generated samples to estimate mutual information, we can potentially extend the learning framework to larger and more complicated neural network architectures.

This work is our first step toward exploiting expressive power of large neural networks using informationtheoretic perspective that is not yet fully utilized.

@highlight

Learning a better neural networks' representation with Information Bottleneck principle

@highlight

Proposes a learning method based on the information bottleneck framework, where hidden layers of deep nets compress the input X while maintaining sufficient information to predict the output Y.

@highlight

This paper presents a new way of training stochastic neural network following an information relevance/compression framework similar to the Information Bottleneck.