Federated learning allows edge devices to collaboratively learn a shared model while keeping the training data on device, decoupling the ability to do model training from the need to store the data in the cloud.

We propose Federated matched averaging (FedMA) algorithm designed for federated learning of modern neural network architectures e.g. convolutional neural networks (CNNs) and LSTMs.

FedMA constructs the shared global model in a layer-wise manner by matching and averaging hidden elements (i.e. channels for convolution layers; hidden states for LSTM; neurons for fully connected layers) with similar feature extraction signatures.

Our experiments indicate that FedMA outperforms popular state-of-the-art federated learning algorithms on deep CNN and LSTM architectures trained on real world datasets, while improving the communication efficiency.

Edge devices such as mobile phones, sensor networks or vehicles have access to a wealth of data.

However, due to concerns raised by data privacy, network bandwidth limitation, and device availability, it's unpractical to gather all local data to the data center and conduct centralized training.

To address these concerns, federated learning is emerging (McMahan et al., 2017; Li et al., 2019; Smith et al., 2017; Caldas et al., 2018; Bonawitz et al., 2019) to allow local clients to collaboratively train a shared global model.

The typical federated learning paradigm involves two stages: (i) clients train models over their datasets independently (ii) the data center uploads their locally trained models.

The data center then aggregates the received models into a shared global model.

One of the standard aggregation methods is FedAvg (McMahan et al., 2017) where parameters of local models are averaged element-wise with weights proportional to sizes of client datasets.

FedProx (Sahu et al., 2018 ) adds a proximal term for client local cost functions, which limits the impact of local updates by restricting them to be close to the global model.

Agnostic Federated Learning (AFL) (Mohri et al., 2019) , as another variant of FedAvg, optimizes a centralized distribution that is formed by a mixture of the client distributions.

One shortcoming of the FedAvg algorithm is that coordinate-wise averaging of weights may have drastic detrimental effect on the performance and hence hinders the communication efficiency.

This issue arises due to the permutation invariant nature of the neural network (NN) parameters, i.e. for any given NN there are many variations of it that only differ in the ordering of parameters and constitute local optima which are practically equivalent.

Probabilistic Federated Neural Matching (PFNM) (Yurochkin et al., 2019) addresses this problem by finding permutation of the parameters of the NNs before averaging them.

PFNM further utilizes Bayesian nonparametric machinery to adapt global model size to heterogeneity of the data.

As a result, PFNM has better performance and communication efficiency, however it was only developed for fully connected NNs and tested on simple architectures.

Our contribution In this work (i) we demonstrate how PFNM can be applied to CNNs and LSTMs, however we find that it gives very minor improvement over weight averaging when applied to modern deep neural network architectures; (ii) we propose Federated Matched Averaging (FedMA), a new layers-wise federated learning algorithm for modern CNNs and LSTMs utilizing matching and model size adaptation underpinnings of PFNM; (iii) We empirically study FedMA with real datasets under the federated learning constraints.

In this section we will discuss permutation invariance classes of prominent neural network architectures and establish the appropriate notion of averaging in the parameter space of NNs.

We will begin with the simplest case of a single hidden layer fully connected network, moving on to deep architectures and, finally, convolutional and recurrent architectures.

Permutation invariance of fully connected architectures A basic fully connected (FC) NN can be formulated asŷ = σ(xW 1 )W 2 (without loss of generality, biases are omitted to simplify notation), where σ is the non-linearity (applied entry-wise).

Expanding the preceding expression

, where i· and ·i denote ith row and column correspondingly and L is the number of hidden units.

Summation is a permutation invariant operation, hence for any {W 1 , W 2 } there are L!

practically equivalent parametrizations if this basic NN.

It is then more appropriate to writeŷ

Recall that permutation matrix is an orthogonal matrix that acts on rows when applied on the left and on columns when applied on the right.

Suppose {W 1 , W 2 } are optimal weights, then weights obtained from training on two homogeneous datasets X j , X j are

It is now easy to see why naive averaging in the parameter space is not appropriate: with non-negligible probability Π j = Π j and (W 1 Π j + W 1 Π j )/2 = W 1 Π for any Π. To meaningfully average neural networks in the weight space we should first undo the permutation

In this section we formulate practical notion of parameter averaging under the permutation invariance.

Let w jl be lth neuron learned on dataset j (i.e. lth column of W

(1) Π j in the previous example), θ i denote the ith neuron in the global model, and c(·, ·) be an appropriate similarity function between a pair of neurons.

Solution to the following optimization problem are the required inverse permutations:

Then Π Solving matched averaging Objective function in equation 2 can be optimized using an iterative procedure: applying the Hungarian matching algorithm (Kuhn, 1955) to find permutation {π j li } l,i corresponding to dataset j , holding other permutations {π j li } l,i,j =j fixed and iterating over the datasets.

Important aspect of Federated Learning that we should consider here is the data heterogeneity.

Every client will learn a collection of feature extractors, i.e. neural network weights, representing their individual data modality.

As a consequence, feature extractors learned across clients may overlap only partially.

To account for this we allow the size of the global model L to be an unknown variable satisfying max j L j ≤ L ≤ j L j where L j is the number of neurons learned from dataset j. That is, global model is at least as big as the largest of the local models and at most as big as the concatenation of all the local models.

Next we show that matched averaging with adaptive global model size remains amendable to iterative Hungarian algorithm with a special cost.

At each iteration, given current estimates of {π j li } l,i,j =j , we find a corresponding global model

(this is typically a closed-form expression or a simple optimization sub-problem, e.g. a mean if c(·, ·) is Euclidean) and then we will use Hungarian algorithm to match this global model to neurons {w j l } L j l=1 of the dataset j to obtain a new global model with L ≤ L ≤ L + L j neurons.

Due to data heterogeneity, local model j may have neurons not present in the global model built from other local models, therefore we want to avoid "poor" matches by saying that if the optimal match has cost larger than some threshold value , instead of matching we create a new global neuron from the corresponding local one.

We also want a modest size global model and therefore penalize its size with some increasing function f (L ).

This intuition is formalized in the following extended maximum bipartite matching formulation:

The size of the new global model is then L = max{i : π j li = 1, l = 1, . . .

, L j }.

We note some technical details: after the optimization is done, each corresponding Π T j is of size L j × L and is not a permutation matrix in a classical sense when L j = L. Its functionality is however similar: taking matrix product with a weight matrix W

(1) j Π T j implies permuting the weights to align with weights learned on the other datasets and padding with "dummy" neurons having zero weights (alternatively we can pad weights W (1) j first and complete Π T j with missing rows to recover a proper permutation matrix).

This "dummy" neurons should also be discounted when taking average.

Without loss of generality, in the subsequent presentation we will ignore these technicalities to simplify the notation.

To complete the matched averaging optimization procedure it remains to specify similarity c(·, ·), threshold and model size penalty f (·).

Although one can consider application specific choices, here for simplicity we follow setup of Yurochkin et al. (2019) .

They arrived at a special case of equation 3 to compute maximum a posteriori estimate (MAP) of their Bayesian nonparametric model based on the Beta-Bernoulli process (BBP) (Thibaux & Jordan, 2007) , where similarity c(w jl , θ i ) is the corresponding posterior probability of jth client neuron l generated from a Gaussian with mean θ i , and and f (·) are guided by the Indian Buffet Process prior (Ghahramani & Griffiths, 2005) .

We refer to a procedure for solving equation 2 with the setup from Yurochkin et al. (2019) as BBP-MAP.

We note that their Probabilistic Federated Neural Matching (PFNM) is only applicable to fully connected architectures limiting its practicality.

Our matched averaging perspective allows to formulate averaging of widely used architectures such as CNNs and LSTMs as instances of equation 2 and utilize the BBP-MAP as a solver.

Before moving onto the convolutional and recurrent architectures, we discuss permutation invariance in deep fully connected networks and corresponding matched averaging approach.

We will utilize this as a building block for handling LSTMs and CNN architectures such as VGG (Simonyan & Zisserman, 2014) widely used in practice.

We extend equation 1 to recursively define deep FC network:

where n = 1, . . . , N is the layer index, Π 0 is identity indicating non-ambiguity in the ordering of input features x = x 0 and Π N is identity for the same in output classes.

Conventionally σ(·) is any non-linearity except forŷ = x N where it is the identity function (or softmax if we want probabilities instead of logits).

When N = 2, we recover a single hidden layer variant from equation 1.

To perform matched averaging of deep FCs obtained from J clients we need to find inverse permutations for every layer of every client.

Unfortunately, permutations within any consecutive pair of intermediate layers are coupled leading to a NP-hard combinatorial optimization problem.

Instead we consider recursive (in layers) matched averaging formulation.

Suppose we have {Π j,n−1 }, then plugging {Π T j,n−1 W j,n } into equation 2 we find {Π j,n } and move onto next layer.

The recursion base for this procedure is {Π j,0 }, which we know is an identity permutation for any j.

Permutation invariance of CNNs The key observation in understanding permutation invariance of CNNs is that instead of neurons, channels define the invariance.

To be more concrete, let Conv(x, W ) define convolutional operation on input x with weights W ∈ R

, where C in , C out are the numbers of input/output channels and w, h are the width and height of the filters.

Applying any permutation to the output dimension of the weights and then same permutation to the input channel dimension of the subsequent layer will not change the corresponding CNN's forward pass.

Analogous to equation 4 we can write:

Note that this formulation permits pooling operations as those act within channels.

To apply matched averaging for the nth CNN layer we form inputs to equation 2 as

This result can be alternatively derived taking the IM2COL perspective.

Similar to FCs, we can recursively perform matched averaging on deep CNNs.

The immediate consequence of our result is the extension of PFNM (Yurochkin et al., 2019) to CNNs.

Empirically (Figure 1 , see One-Shot Matching) we found that this extension performs well on MNIST with a simpler CNN architecture such as LeNet (LeCun et al., 1998) (4 layers) and significantly outperforms coordinate-wise weight averaging (1 round FedAvg).

However, it breaks down for more complex architecture, e.g. VGG-9 (Simonyan & Zisserman, 2014) (9 layers), needed to obtain good quality prediction on a more challenging CIFAR-10.

Permutation invariance of LSTMs Permutation invariance in the recurrent architectures is associated with the ordering of the hidden states.

At a first glance it appears similar to fully connected architecture, however the important difference is associated with the permutation invariance of the hidden-to-hidden weights H ∈ R L×L , where L is the number of hidden states.

In particular, permutation of the hidden states affects both rows and columns of H. Consider a basic RNN h t = σ(h t−1 H + x t W ), where W are the input-to-hidden weights.

To account for the permutation invariance of the hidden states, we notice that dimensions of h t should be permuted in the same way for any t, hence

To match RNNs, the basic sub-problem is to align hidden-to-hidden weights of two clients with Euclidean similarity, which requires minimizing Π T H j Π − H j 2 2 over permutations Π.

This is a quadratic assignment problem, which is NP-hard.

Fortunately, the same permutation appears in an already familiar context of input-to-hidden matching of W Π. Our matched averaging RNN solution is to utilize equation 2 plugging-in input-to-hidden weights {W j } to find {Π j }.

Then federated hidden-to-hidden weights are computed as H = 1 J j Π j H h Π T j and input-to-hidden weights are computed as before.

LSTMs have multiple cell states, each having its individual hiddento-hidden and input-to-hidden weights.

In out matched averaging we stack input-to-hidden weights into SD × L weight matrix (S is the number of cell states; D is input dimension and L is the number of hidden states) when computing the permutation matrices and then average all weights as described previously.

LSTMs also often have an embedding layer, which we handle like a fully connected layer.

Finally, we process deep LSTMs in the recursive manner similar to deep FCs.

Defining the permutation invariance classes of CNNs and LSTMs allows us to extend PFNM (Yurochkin et al., 2019) to these architectures, however our empirical study in Figure 1 (see OneShot Matching) demonstrates that such extension fails on deep architectures necessary to solve more complex tasks.

Our results suggest that recursive handling of layers with matched averaging may entail poor overall solution.

To alleviate this problem and utilize the strength of matched averaging on "shallow" architectures, we propose the following layer-wise matching scheme.

First, data center gathers only the first layer's weights from the clients and performs one-layer matching described previously to obtain the first layers weights of the federated model.

Data center then broadcasts these weights to the clients, which proceed to train all consecutive layers on their datasets, keeping the matched federated layers frozen.

This procedure is then repeated up to the last layer for which we conduct a weighted averaging based on the class proportions of data points per client.

We summarize our Federated Matched Averaging (FedMA) in Algorithm 1.

The FedMA approach requires communication rounds equal to the number of layers in a network.

In Figure 1 we show that with layer-wise matching FedMA performs well on the deeper VGG-9 CNN as well as LSTMs.

In the more challenging heterogeneous setting, FedMA outperforms FedAvg, FedProx trained with same number of communication rounds (4 for LeNet and LSTM and 9 for VGG-9) and other baselines, i.e. client individual CNNs and their ensemble.

j p jk W jl,n where p k is fraction of data points with label k on worker j; end for j ∈ {1, . . .

, J} do W j,n+1 ← Π T j,n W j,n+1 ; // permutate the next-layer weights Train {W j,n+1 , . . .

, W j,L } with W n frozen; end n = n + 1; end FedMA with communication We've shown that in the heterogeneous data scenario FedMA outperforms other federated learning approaches, however it still lags in performance behind the entire data training.

Of course the entire data training is not possible under the federated learning constraints, but it serves as performance upper bound we should strive to achieve.

To further improve the performance of our method, we propose FedMA with communication, where local clients receive the matched global model at the beginning of a new round and reconstruct their local models with the size equal to the original local models (e.g. size of a VGG-9) based on the matching results of the previous round.

This procedure allows to keep the size of the global model small in contrast to a naive strategy of utilizing full matched global model as a starting point across clients on every round.

We present an empirical study of FedMA with communication and compare it with state-of-the-art methods i.e. FedAvg (McMahan et al., 2017) and FedProx (Sahu et al., 2018) ; analyze the performance under the growing number of clients and visualize the matching behavior of FedMA to study its interpretability.

Our experimental studies are conducted over three real world datasets.

Summary information about the datasets and associated models can be found in supplement Table 3 .

Experimental Setup We implemented FedMA and the considered baseline methods in PyTorch (Paszke et al., 2017) .

We deploy our empirical study under a simulated federated learning environment where we treat one centralized node in the distributed cluster as the data center and the other nodes as local clients.

All nodes in our experiments are deployed on p3.2xlarge instances on Amazon EC2.

We assume the data center samples all the clients to join the training process for every communication round for simplicity.

For the CIFAR-10 dataset, we use data augmentation (random crops, and flips) and normalize each individual image (details provided in the Supplement).

We note that we ignore all batch normalization (Ioffe & Szegedy, 2015) layers in the VGG architecture and leave it for future work.

For CIFAR-10, we considered two data partition strategies to simulate federated learning scenario: (i) homogeneous partition where each local client has approximately equal proportion of each of the classes; (ii) heterogeneous partition for which number of data points and class proportions are unbalanced.

We simulated a heterogeneous partition into J clients by sampling p k ∼ Dir J (0.5) and allocating a p k,j proportion of the training instances of class k to local client j.

We use the original test set in CIFAR-10 as our global test set and all test accuracy in our experiments are conducted over that test set.

For the Shakespeare dataset, since each speaking role in each play is considered a different client according to Caldas et al. (2018) , it's inherently heterogeneous.

We preprocess the Shakespeare dataset by filtering out the clients with datapoints less 10k and get 132 clients in total.

We choose 80% of data in training set.

We then randomly sample J = 66 out of 132 clients in conducting our experiments.

We amalgamate all test sets on clients as our global test set.

In this experiment we study performance of FedMA with communication.

Our goal is to compare our method to FedAvg and FedProx in terms of the total message size exchanged between data center and clients (in Gigabytes) and the number of communication rounds (recall that completing one FedMA pass requires number of rounds equal to the number of layers in the local models) needed for the global model to achieve good performance on the test data.

We also compare to the performance of an ensemble method.

We evaluate all methods under the heterogeneous federated learning scenario on CIFAR-10 with J = 16 clients with VGG-9 local models and on Shakespeare dataset with J = 66 clients with 1-layer LSTM network.

We fix the total rounds of communication allowed for FedMA, FedAvg, and FedProx i.e. 11 rounds for FedMA and 99/33 rounds for FedAvg and FedProx for the VGG-9/LSTM experiments respectively.

We notice that local training epoch is a common parameter shared by the three considered methods, we thus tune the local training epochs (we denote it by E) (comprehensive analysis will be presented in the next experiment) and report the convergence rate under the best E that yields the best final model accuracy over the global test set.

We also notice that there is another hyper-parameter in FedProx i.e. the coefficient µ associated with the proxy term, we also tune the parameter using grid search and report the best µ we found i.e. 0.001 for both VGG-9 and LSTM experiments.

FedMA outperforms FedAvg and FedProx in all scenarios (Figure 2 ) with its advantage especially pronounced when we evaluate convergence as a function of the message size in Figures  2(a) and 2(c) .

Final performance of all trained models is summarized in Tables 1 and 2 .

Effect of local training epochs As studied in previous work (McMahan et al., 2017; Caldas et al., 2018; Sahu et al., 2018) , the number of local training epochs E can affect the performance of FedAvg and sometimes lead to divergence.

We conduct an experimental study on the effect of E over FedAvg, FedProx, and FedMA on VGG-9 trained on CIFAR-10 under heterogeneous setup.

The candidate local epochs we considered are E ∈ {10, 20, 50, 70, 100, 150}. For each of the candidate E, we run FedMA for 6 rounds while FedAvg and FedProx for 54 rounds and report the final accuracy that each methods achieves.

The result is shown in Figure 3 .

We observed that training longer favors the convergence rate of FedMA, which matches the our assumption that FedMA returns better global model on local models with higher quality.

For FedAvg, longer local training leads to deterioration of the final accuracy, which matches the observation in the previous literature (McMahan et al., 2017; Caldas et al., 2018; Sahu et al., 2018) .

FedProx prevents the accuracy deterioration to some extent, however, the accuracy of final model still gets reduced.

The result of this experiment suggests that FedMA is the only method that local clients can use to train their model as long as they want.

Handling data bias Real world data often exhibit multimodality within each class, e.g. geodiversity.

It has been shown that an observable amerocentric and eurocentric bias is present in the widely used ImageNet dataset (Shankar et al., 2017; Russakovsky et al., 2015) .

Classifiers trained on such data "learn" these biases and perform poorly on the under-represented domains (modalities) since correlation between the corresponding dominating domain and class can prevent the classifier from learning meaningful relations between features and classes.

For example, classifier trained on amerocentric and eurocentric data may learn to associate white color dress with a "bride" class, therefore underperforming on the wedding images taken in countries where wedding traditions are different (Doshi, 2018) .

The data bias scenario is an important aspect of federated learning, however it received little to no attention in the prior federated learning works.

In this study we argue that FedMA can handle this type of problem.

If we view each domain, e.g. geographic region, as one client, local models will not be affected by the aggregate data biases and learn meaningful relations between features and classes.

FedMA can then be used to learn a good global model without biases.

We have already demonstrated strong performance of FedMA on federated learning problems with heterogeneous data across clients and this scenario is very similar.

To verify this conjecture we conduct the following experiment.

We simulate the skewed domain problem with CIFAR-10 dataset by randomly selecting 5 classes and making 95% training images in those classes to be grayscale.

For the remaining 5 we turn only 5% of the corresponding images into grayscale.

By doing so, we create 5 grayscale images dominated classes and 5 colored images dominated classes.

In the test set, there is half grayscale and half colored images for each class.

We anticipate entire data training to pick up the uninformative correlations between greyscale and certain classes, leading to poor test performance without these correlations.

In Figure 4 we see that entire data training performs poorly in comparison to the regular (i.e. No Bias) training and testing on CIFAR-10 dataset without any grayscaling.

This experiment was motivated by Olga Russakovsky's talk at ICML 2019.

Next we compare the federated learning based approaches.

We split the images from color dominated classes and grayscale dominated classes into 2 clients.

We then conduct FedMA with communication, FedAvg, and FedProx with these 2 clients.

FedMA noticeably outperforms the entire data training and other federated learning approach as shown in Figure 4 .

This result suggests that FedMA may be of interest beyond learning under the federated learning constraints, where entire data training is the performance upper bound, but also to eliminate data biases and outperform entire data training.

We consider two additional approaches to eliminate data bias without the federated learning constraints.

One way to alleviate data bias is to selectively collect more data to debias the dataset.

In the context of our experiment, this means getting more colored images for grayscale dominated classes and more grayscale images for color dominated classes.

We simulate this scenario by simply doing a full data training where each class in both train and test images has equal amount of grayscale and color images.

This procedure, Color Balanced, performs well, but selective collection of new data in practice may be expensive or even not possible.

Instead of collecting new data, one may consider oversampling from the available data to debias.

In Oversampling, we sample the underrepresented domain (via sampling with replacement) to make the proportion of color and grayscale images to be equal for each class (oversampled images are also passed through the data augmentation pipeline, e.g. random flipping and cropping, to further enforce the data diversity).

Such procedure may be prone to overfitting the oversampled images and we see that this approach only provides marginal improvement of the model accuracy compared to centralized training over the skewed dataset and performs noticeably worse than FedMA.

Data efficiency It is known that deep learning models perform better when more training data is available.

However, under the federated learning constraints, data efficiency has not been studied to the best of our knowledge.

The challenge here is that when new clients join the federated system, they each bring their own version of the data distribution, which, if not handled properly, may deteriorate the performance despite the growing data size across the clients.

To simulate this scenario we first partition the entire training CIFAR-10 dataset into 5 homogeneous pieces.

We then partition each homogeneous data piece further into 5 sub-pieces heterogeneously.

Using this strategy, we partition the CIFAR-10 training set into 25 heterogeneous small sub-datasets containing approximately 2k points each.

We conduct a 5-step experimental study: starting from a randomly selected homogeneous piece consisting of 5 associated heterogeneous sub-pieces, we simulate a 5-client federated learning heterogeneous problem.

For each consecutive step, we add one of the remaining homogeneous data pieces consisting of 5 new clients with heterogeneous sub-datasets.

Results are presented in Figure 5 .

Performance of FedMA (with a single pass) improves when new clients are added to the federated learning system, while FedAvg with 9 communication rounds deteriorates.

Interpretability One of the strengths of FedMA is that it utilizes communication rounds more efficiently than FedAvg.

Instead of directly averaging weights element-wise, FedMA identifies matching groups of convolutional filters and then averages them into the global convolutional filters.

It's natural to ask "How does the matched filters look like?".

In Figure 6 we visualize the representations generated by a pair of matched local filters, aggregated global filter, and the filter returned by the FedAvg method over the same input image.

Matched filters and the global filter found with FedMA are extracting the same feature of the input image, i.e. filter 0 of client 1 and filter 23 of client 2 are extracting the position of the legs of the horse, and the corresponding matched global filter 0 does the same.

For the FedAvg, global filter 0 is the average of filter 0 of client 1 and filter 0 of client 2, which clearly tampers the leg extraction functionality of filter 0 of client 1.

In this paper, we presented FedMA, a new layer-wise federated learning algorithm designed for modern CNNs and LSTMs architectures utilizing probabilistic matching and model size adaptation.

We demonstrate the convergence rate and communication efficiency of FedMA empirically.

In the future, we would like to extend FedMA towards finding the optimal averaging strategy.

Making FedMa support more building blocks e.g. residual structures in CNNs and batch normalization layers is also of interest.

Table 4 : Detailed information of the VGG-9 architecture used in our experiments, all non-linear activation function in this architecture is ReLU; the shapes for convolution layers follows (Cin, Cout, c, c) In preprocessing the images in CIFAR-10 dataset, we follow the standard data augmentation and normalization process.

For data augmentation, random cropping and horizontal random flipping are used.

Each color channels are normalized with mean and standard deviation by µ r = 0.491372549, µ g = 0.482352941, µ b = 0.446666667, σ r = 0.247058824, σ g = 0.243529412, σ b = 0.261568627.

Each channel pixel is normalized by subtracting the mean value in this color channel and then divided by the standard deviation of this color channel.

Here we report the shapes of final global VGG and LSTM models returned by FRB with communication.

<|TLDR|>

@highlight

Communication efficient federated learning with layer-wise matching