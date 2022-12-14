We propose a novel framework to adaptively adjust the dropout rates for the deep neural network based on a Rademacher complexity bound.

The state-of-the-art deep learning algorithms impose dropout strategy to prevent feature co-adaptation.

However, choosing the dropout rates remains an art of heuristics or relies on empirical grid-search over some hyperparameter space.

In this work, we show the network Rademacher complexity is bounded by a function related to the dropout rate vectors and the weight coefficient matrices.

Subsequently, we impose this bound as a regularizer and provide a theoretical justified way to trade-off between model complexity and representation power.

Therefore, the dropout rates and the empirical loss are unified into the same objective function, which is then optimized using the block coordinate descent algorithm.

We discover that the adaptively adjusted dropout rates converge to some interesting distributions that reveal meaningful patterns.

Experiments on the task of image and document classification also show our method achieves better performance compared to the state-of the-art dropout algorithms.

Dropout training BID19 has been proposed to regularize deep neural networks for classification tasks.

It has been shown to work well in reducing co-adaptation of neurons-and hence, preventing model overfitting.

The idea of dropout is to stochastically set a neuron's output to zero according to Bernoulli random variables.

It has been a crucial component in the winning solution to visual object recognition on ImageNet BID7 .

Ever since, there have been many follow-ups on novel learning algorithms (Goodfellow et al., 2013; BID1 , regularization techniques BID23 , and fast approximations BID25 .However, the classical dropout model has a few limitations.

First, the model requires to specify the retain rates, i.e., the probabilities of keeping a neuron's output, a priori to model training.

Subsequently, these retain rates are kept fixed throughout the training process thereafter.

It is often not clear how to choose the retain rates in an optimal way.

They are usually set via grid-search over hyper-parameter space or simply according to some rule-of-thumb.

Another limitation is that all neurons in the same layer share the same retain rate.

This exponentially reduces the search space of hyper-parameter optimization.

For example, BID19 use a fixed retain probability throughout training for all dropout variables in each layer.

In this paper, we propose a novel regularizer based on the Rademacher complexity of a neural network BID16 .

Without loss of generality, we use multilayer perceptron with dropout as our example and prove its Rademacher complexity is bounded by a term related to the dropout probabilities.

This enables us to explicitly incorporate the model complexity term as a regularizer into the objective function.

This Rademacher complexity bound regularizer provides us a lot of flexibility and advantage in modeling and optimization.

First, it combines the model complexity and the loss function in an unified objective.

This offers a viable way to trade-off the model complexity and representation power through the regularizer weighting coefficient.

Second, since this bound is a function of dropout probabilities, we are able to incorporate them explictly into the computation graph of the optimization procedure.

We can then adaptively optimize the objective and adjust the dropout probabilities throughout training in a way similar to ridge regression and the lasso BID4 .

Third, our proposed regularizer assumes a neuron-wise dropout manner and models different neurons to have different retain rates during the optimization.

Our empirical results demonstrate interesting trend on the changes in histograms of dropout probabilities for both hidden and input layers.

We also discover that the distribution over retain rates upon model convergence reveals meaningful pattern on the input features.

To the best of our knowledge, this is the first ever effort of using the Rademacher complexity bound to adaptively adjust the dropout probabilities for the neural networks.

We organize the rest of the paper as following.

Section 2 reviews some past approaches well aligned with our motivation, and highlight some major difference to our proposed approach.

We subsequently detail our proposed approach in Section 3.

In Section 4, we present our thorough empirical evaluations on the task of image and document classification on several benchmark datasets.

Finally, Section 5 concludes this paper and summarizes some possible future research ideas.

There are several prior works well aligned with our motivation and addressing similar problems, but significantly different from our method.

For example, the standout network BID0 extends dropout network into a complex network structure, by interleaving a binary belief network with a regular deep neural network.

The binary belief network controls the dropout rate for each neuron, backward propagates classification error and adaptively adjust according to training data.

BID27 realize the dropout training via the concept of Bayesian feature noising during neural network learning.

They further extend the model to incorporate either dimension-specific or group-specific noise and propose framework to adaptively learn the dropout rates.

sample dropout from a multinomial distribution on neuron basis and establish a risk bound for stochastic optimization.

They then propose the evolutional dropout model to adaptively update the sampling probabilities during training time.

In addition to these approaches, one other family of solution is via the concept of regularizer.

BID25 propose fast approximation methods to marginalize the dropout layer and show that the classical dropout can be approximated by a Gaussian distribution.

Later, BID23 show that the dropout training on generalized linear models can be viewed as a form of adaptive regularization technique.

Gal & Ghahramani (2016) develop a new theoretical framework casting dropout training as approximation to Bayesian inference in deep Gaussian processes.

It also provides a theoretical justification and formulates dropout into a special case of Bayesian regularization.

In the mean time, Maeda (2014) discusses a Bayesian perspective on dropout focusing on the binary variant, and also demonstrate encourage experimental results.

Generalized dropout BID18 further unifies the dropout model into a rich family of regularizers and propose a Bayesian approach to update dropout rates.

One popular method along with these works is the variational dropout method BID6 , which provides an elegant interpretation of Gaussian dropout as a special case of Bayesian regularization.

It also proposes a Bayesian inference method using a local reparameterization technique and translates uncertainty of global parameters into local noise.

Hence, it allows inference on the parameterized Bayesian posteriors for dropout rates.

This allows us to adaptively tune individual dropout rates on layer, neuron or even weight level in a Bayesian manner.

Recently, BID13 extend the variational dropout method with a tighter approximation which subsequently produce more sparse dropout rates.

However, these models are fundamentally different than our proposed approach.

They directly operates on the Gaussian approximation of dropout models rather than the canonical multiplicative dropout model, whereas our proposed method directly bounds the model complexity of classical dropout model.

Meanwhile, the model complexity and the generalization capability of deep neural networks have been well studied in theoretical perspective.

BID24 prove the generalization bound for the DropConnect neural networks-a weight-wise variant of dropout model.

Later, Gao & Zhou (2016) extend the work and derive a Rademacher complexity bound for deep neural networks with dropout.

Another perspective to the model generalization is the PAC-Bayes bound proposed by BID12 .

The PAC-Bayes method assumes probability measures on the hypothesis space, and gives generalization guarantee over all possible "priors".

BID12 give a PAC-Bayes bound for linear predictors with dropout.

In practise, the PAC-Bayes method has the potential to give a even tigher generalization bound.

The bound we prove in this paper is based on traditional techniques using Rademacher complexity.

It is a first step towards understanding how the dropout method works, and we would like to extend it to the PAC-Bayes paradigm in the future.

These works provide a theoretical guarantee and mathematical justification on the effectiveness of dropout method in general.

However, they all assume that all input and hidden layers have the same dropout rates.

Thus their bound can not be applied to our algorithm.

We would like to focus on the classification problem and use multilayer perceptron as our example.

However, note that the similar idea could be easily extended to general feedforward networks.

Let us assume a labeled dataset DISPLAYFORM0 DISPLAYFORM1 For an input sample feature vector x ??? R d , the function before the activation of the j th neuron in the DISPLAYFORM2 where ?? : R ??? R + is the rectified linear activation function (Nair & Hinton, 2010, ReLU) .

In vector form, if we denote as the Hadamard product, we could write the output of the l th layer as DISPLAYFORM3 Without loss of generality, we also apply Bernoulli dropout to the input layer parameter DISPLAYFORM4 Note that the output of the neural network f L (x; W, r) ??? R k is a random vector due to the Bernoulli random variables r.

We use the expected value of f L (x; W, r) as the deterministic output DISPLAYFORM5 The final predictions are made through a softmax function, and we use the cross-entropy loss as our optimization objective.

To simplify our analysis, we follow BID24 and reformulate the cross-entropy loss on top of the softmax into a single logistic function DISPLAYFORM6 .

Definition The empirical Rademacher complexity of function class F with respect to the sample S is DISPLAYFORM0 Define loss ??? f L as the composition of the logistic loss function loss and the neural function f L returned from the L th (last) layer, i.e., DISPLAYFORM1 Theorem 3.1.

Let X ??? R n??d be the sample matrix with the i th row DISPLAYFORM2 . .

, L} , given ??, the empirical Rademacher complexity of the loss for the dropout neural network defined above is bounded by DISPLAYFORM3 where k is the number of classes to predict, ?? l is the k l -dimensional vector of Bernoulli parameters for the dropout random variables in the l th layer, ?? i s are i.i.d.

Rademacher random variables, and ?? max is the matrix max norm defined as A max = max ij |A ij |.

We observe that the empirical loss and Rademacher regularizer change roughly in a monotonic way as a function of retain rates on training data.

The experiments are evaluated on MNIST dataset with a hidden layer of 128 ReLU units.

We apply dropout on the hidden layer only, and keep the retain rates fixed throughout training.

We optimize with the empirical loss Loss(S, f L (??; W, ??)), i.e., without any regularizer for 200 epochs with minibatch fo 100.

All Rademacher regularizers are computed after every epoch in post-hoc manner, under the settings of p = ???, q = 1.

We plot the samples from last 20 epochs under each settings, with initial learning rate of 0.01, and decay by half every 40 epochs.

Please refer to the appendix for the proof.

Theorem 3.1 suggests that the empirical Rademacher complexity of the dropout network specified in this paper is related to several terms in a multiplicative way:i: p-norms of the coefficients: DISPLAYFORM4 Note that in BID19 , 2 norms of the coefficients are already used as regularizers in the experimental comparison ii: 1-norms of the retain rates ?? l iii: sample related metrics: dimension of the sample d, the number of samples n, and maximum entries in the samples X iv: the number of classes in the prediction kAn the extreme case is, if the retain rates ?? l for one layer are all zeros, then the upper bound above is tight, since in this case the network is simply doing random guess for predictions.

Similarly when the coefficients in one layer are all zeros, the bound is also tight.

In both cases the features from the samples are not even used in the prediction due to either zero retain rates or zero coefficients.

We have shown that the Rademacher complexity of a neural network is bounded by a function of the dropout rates, i.e., Bernoulli parameters ??.

This makes it possible to unify the dropout rates and the network coefficients W in one objective.

By imposing our upper bound of Rademacher complexity to the loss function as a regularizer, we have DISPLAYFORM0 where the variable ?? ??? R + is a weighting coefficient to trade off the training loss and the generalization capability.

The empirical loss Loss(S, f L (??; W, ??)) and regularizer function Reg(S, W, ??) are defined as DISPLAYFORM1 where W l j is the j th column of W l and ?? l is the retain rate vector for the l th layer.

The variable k is the number of classes to predict and X ??? R n??d is the sample matrix.

In addition to the Rademacher regularizer Reg(S, W, ??), the empirical loss term Loss(S, f L (??; W, ??)) also depends on the dropout Bernoulli parameters ??.

Intuitively, when ?? becomes smaller, the loss term Loss(S, f L (??; W, ??)) becomes larger, since the model is less capable to fit the training samples (i.e., less representation power), the empirical Rademacher complexity bound becomes smaller (i.e., more generalizable), and vice versa.

FIG0 plots the cross-entropy loss and empirical Rademacher p = ???, q = 1 regularizer upon model convergence under different settings of retain rates.

In the extreme case, when all ?? l j become zeros, the model always makes random guess for prediction, leading to a large fitness error Loss(S, f L (??; W, ??)), and the Rademacher complexity ReLU in stochastic mode 1x1024 ReLU in deterministic mode 2x800 ReLU in stochastic mode 2x800 ReLU in deterministic mode 3x1024 ReLU in stochastic mode 3x1024 ReLU in deterministic mode Figure 2 : Changes in the true objectives in "stochastic" mode and their "deterministic" approximations against training epochs under different network architectures.

The optimization objectives are reported on the training set of MNIST dataset, with Rademacher regularizer.

We use minibatch size of 100, initial learning rate of 0.01, and decay it by half every 200 epochs.

The network structures we evaluated includes 1 hidden layer with 1024 units, 2 hidden layers with 800 units each, and 3 hidden layers with 1024 units each.

The regularizer weights are set to 1e ???3 , 1e ???4 and 1e ???5 respectively.

All neurons are ReLU units.

Empirically, we find that optimizing the "deterministic" objective leads to similar improvements on the true "stochastic" objective as in Eqn.

(2).

DISPLAYFORM2 We now incorporate the Bernoulli parameters ?? into the optimization objective as in Eqn.

FORMULA12 , i.e., the objective is a function of both weight coefficient matrices W and retain rate vectors ??.

In particular, the model parameters and the dropout rates are optimized using a block coordinate descent algorithm.

We start with an initial setting of W and ??, and optimize W and ?? in an alternating fashion.

For the retain rate probability ??, due to the stochastic nature of dropout framework, it is very expensive to compute the exact objective value.

For each training instance, we may have to exhaustively enumerate all possible dropout configurations in a combinatorial search space and compute its expectation of all the objective functions.

One possbile approximation is to iteratively taking large number of samples from Bernoulli distributions of all layers for any input data and then compute the average objective.

Even though, the computational complexity can be exponential as to the number of training data.

Therefore, in our case, during the optimization of ??, we use the expected value of the Bernoulli dropout variables to rescale the output from each layer, to approximate the true objective f L (x; W, ??).

It significantly speeds up the forward propagation process, as we do not need to iteratively sample the dropout variables for each training example.

Essentially, it makes the layer output deterministic and the underlying network operates as if without dropout, which is exactly the same as the approximation used in BID19 during testing time.

Note that using the expected value of Bernoulli dropout random variables to rescale a layer output is an approximation to the true objective f L (x; W, ??).

In practice, we find such "deterministic" approximation exhibits similar behavior during model optimization, and hence does not deviate or alter the performance, but significantly improves the running time.

Figure 2 shows the true objective in "stochastic" mode and its "deterministic" approximation on the training set during optimization process under different network architectures.

Empirically, we observe that the true optimization objective in stochastic mode as in Eqn.

FORMULA12 decreases consistently if we use the expected value of the Bernoulli dropout random variable to approximate the sampling process.

Bottom-Left: sample images from MNIST dataset.

Bottom-Right: retain rates for corresponding input pixels upon model convergence.

The surrounding pixels of input image yield smaller retain rates (corresponds to the dark background area), and the center ones have significantly larger retain rates (corresponds to the number pixels).

We apply our proposed approach with different network architectures, on the task of image and text classification using several public available benchmark datasets.

All hidden neurons and convolutional filters are rectified linear units (Nair & Hinton, 2010, ReLU) .

We found that our approach achieves superior performance against strong baselines on all datasets.

For all datasets, we hold out 20% of the training data as validation set for parameter tuning and model selection.

After then, we combine both of these two sets to train the model and report the classification error rate on test set.

We optimize categorical cross-entropy loss on predicted class labels with Rademacher regularization.

In the context of this paper, we specifically refer the Rademacher regularizer to the case of p = ???, q = 1 unless stated otherwise.

We update the parameters using mini-batch stochastic gradient descent with Nesterov momentum of 0.95 BID20 .For Rademacher complexity term, we perform a grid search on the regularization weight ?? ??? {0.05, 0.01, 0.005, 0.001, 1e ???4 , 1e ???5 }, and update the dropout rates after every I ??? {1, 5, 10, 50, 100} minibatches.

For variational dropout method (Kingma et al., 2015, VARDROP), we examine the both Type-A and Type-B variational dropout with per-layer, per-neuron or per-weight adaptive dropout rate.

We found the neuron-wise adaptive regularization on Type-A variational dropout layer often reports the best performance under most cases.

We also perform a grid search on the regularization noise parameter in {0.1, 0.01, 0.001, 1e???4 , 1e ???5 , 1e ???6 }.

For sparse variational dropout method (Molchanov et al., 2017, SPARSEVARDROP) , we find the model is much more sensitive to regularization weights, and often gets diverged.

We examine different regularization weight in {1e ???3 , 1e ???4 , 1e ???5 }.

We follow similar weight adjustment scheme and scale it up by 10 after first {100, 200, 300} epochs, then further scale up by 5 and 2 after same number of epoch.

In practice, we want to stablize regularization term within some managable variance, so its value does not vary significantly upon difference structure of the underlying neural networks.

Hence, we design some heuristics to scale the regularizer to offset the multipler effects raised from network structure.

For instance, recall the neural network defined in Section 3, the Rademacher complexity regularizer with p = ???, q = 1 after scaling is DISPLAYFORM0 where W l j is the j th column of the weight coefficient matrix W l and ?? l is the retain rate vector for the l th layer.

The variable k is the number of classes to predict and X ??? R n??d is the sample matrix.

Similarly, we could rescale the Rademacher complexity regularizers under other settings of p = 2, q = 2.

Please refer to the appendix for the scaled Rademacher complexity bound regularizers and detailed derivations.

MNIST dataset is a collection of 28 ?? 28 pixel hand-written digit images in grayscale, containing 60K for training and 10K for testing.

The task is to classify the images into 10 digit classes from 0 to 9.

All images are flattened into 784 dimension vectors, and all pixel values are rescaled to gray scale.

We examine several different network structures, including architectures of 1 hiddel layer with 1024 units, 2 hidden layers with 800 neurons each, as well as 3 hidden layers with 1024 units each.

Table 1 compares the performance of our proposed models against other techniques.

We use a learning rate of 0.01 and decay it by 0.5 after every {300, 400, 500} epochs.

We let all models run sufficiently long with 100K updates.

For all models, we also explore different initialization for neuron retaining rates, including {0.8, 1.0} for input layers, {0.5, 0.8, 1.0} for hidden layers.

In practice, we find initializing the retaining rates to 0.8 for input layer and 0.5 for hidden layer yields better performance for all models, except for SPARSE-VARDROP model, initializing retaining rate to 1.0 for input layer seems to give better result.

FIG2 illustrates the changes in retain rates for both input and hidden layers under Rademacher regularization with 1e ???4 regularization weight.

The network contains one hidden layer of 1024 ReLU units.

The retain rates were initialized to 0.8 for input layer and 0.5 for hidden layer.

The learning rate is 0.01 and decayed by half after every 200 epochs.

We observe the retain rates for all layers are diffused throughout training process, and finally converged towards a bimodal distribution for input layer and a unimodal distribution for hidden layer.

We also notice that the retain rates for input layer upon model convergence demonstrate interesting feature pattern of the dataset.

For example, the pixels in surrounding margins yield smaller retain rates, and the center pixels often have larger retain rates.

This is because the digits in MNIST dataset are often centered in the image, hence all the surrounding pixels are not predictive at all when classifying an instance.

This demonstrates that our proposed method is able to dynamically determine if an input signal is informational or not, and subsequently gives higher retain rate if it is, otherwise reduce the retain rate over time.

CIFAR10 and CIFAR100 datasets are collections of 50K training and 10K testing RGB images from 10 and 100 different image categories.

Every instance consists of 32 ?? 32 RGB pixels.

We preprocess all images by subtracting the per-pixel mean computed over all training set, then with ZCA whitening as suggested in BID19 .

No data augmentation is used.

The neural network architecture we evaluate on uses three convolutional layers, each of which followed by a max-pooling layer.

The convolutional layers have 96, 128, and 256 filters respectively.

Each convolutional layer has a 5??5 receptive field applied with a stride of 1 pixel, and each max-pooling layer pools from 3??3 pixel region with strides of 2 pixels.

These convolutional layers are followed by two fully-connected layer having 2048 hidden units each.

Table 2 summarizes the performance of our proposed models against other baselines.

We initialize dropout rates settings with {0.9, 1.0} for input layers, {0.75, 1.0} for convolutional layers and {0.5, 1.0} for fully-connected layers.

Similary to the MNIST evaluation, we find setting the corresponding retaining probabilities for input layers, convolutional layers and fullyconnected layers to 0.9, 0.75 and 0.5 respectively yields best performance under all models.

We initialize the learning rate to 0.001 and decay it exponentially every {200, 300, 400} epochs.

Figure 4 illustrates the changes in retain rates for both input and hidden layers under Rademacher regularization with 0.1 regularization weight.

The network contains two convolution layers with 32 and 64 convolutional filters followed by one fully-connected layer with 1024 neurons.

All hidden units use ReLU activation functions.

The retain rates were initialized to 0.9 for input layer, 0.75 for convolutional layer and 0.5 for fully-connected layer.

The learning rate is 0.001 and exponentially decayed by half after every 300 epochs.

Similar to MNIST dataset, we observe the retain rates for all layers are diffused throughout training process, and finally converged towards a unimodal distribution.

However, unlike MNIST dataset, we do not see similar pattern for retain rates of input layer.

This is mainly due to the nature of dataset, such that CIFAR10 images spread over the entire range, hence all pixels are potentially informational to the classification process.

This again demonstrates that the Rademacher regularizer is able to distinguish the informational pixels and retain them during training.

In addition, we also compare our proposed approach on text classification datasets-SUBJ and IMDB.

SUBJ is a dataset containing 10K subjective and objective sentences BID15 with nearly 14.5K vocabulary after stemming.

All subjective comments come from movie reviews expressing writer's opinion, whereas objective sentences are from movie plots expressing purely facts.

We randomly sample 20% from the collections as test data, and use other 80% for training and validation.

IMDB is a collection of movie reviews from IMDB website, with 25K for training and another 25K for test BID10 , containing more than 50K vocabulary after stemming.

It contains an even number of positive (i.e., with a review score of 7 or more out of a scale of 10) and negative (i.e., with a review score of 4 or less out of 10) reviews.

The dataset has a good movie diversity coverage Figure 4: Changes in retain rates with Rademacher regularization on CIFAR10 dataset.

Top-Left: changes in retain rate histograms for input layer (32 ?? 32 ?? 3 RGB pixels) through training.

TopMiddle: changes in retain rate histograms for first convolutional layer (32 ?? 15 ?? 15 units) through training process.

Top-Right: changes in retain rate histograms for second convolutional layer (64 ?? 7 ?? 7 units) through training process.

Bottom-Left: changes in retain rate histograms for fullyconnected layer (1024 ReLU units) through training process.

Bottom-Middle: sample images from CIFAR10 dataset.

Bottom-Right: retain rates for corresponding input pixels in both superposition and individual RGB channels upon model convergence.

Unlike MNIST datasets, there is no clear pattern from the retain rates out of these channel pixels, since they are all informational towards prediction.with less than 30 reviews per movie.

For each sentence or document in these datasets, we normalize it into a vector of probability distribution over all vocabulary.

Table 3 summarizes the performance of our proposed models against other baselines.

We initialize dropout rates settings with {0.8, 1.0} for input layers and {0.5, 1.0} for fully-connected layers.

Similarly, by setting the corresponding retaining probabilities for input layers and fully-connected layers to 0.8 and 0.5 respectively, the model often yields the best performance.

We use a constant learning rate of 0.001, as well as an initialization learning rate of 0.01 and decay it by half every {200, 300, 400} epochs.

We notice that overall the improvement of dropout is not as significant as MNIST Table 3 : Classification error on text dataset.

FIG4 illustrates the changes in retain rates for both input and hidden layers under Rademacher regularization with 0.005 regularization weight on IMDB dataset.

The network contains one hidden layer of 1024 ReLU units.

The retain rates were initialized to 0.8 for input layer and 0.5 for hidden layer.

The learning rate is 0.01 and decayed by half after every 200 epochs.

Similar to other datasets, we observe the retain rates for all layers are diffused slightly upon model convergence, and in particularly the retain rates for input layer demonstrate interesting feature patterns of the data.

Recall that the task for IMDB dataset is to classify movie reviews into negative or positive labels.

Generically speaking, adjectives are more expressive than nouns or verbs in this scenario, and our findings seems to be consistent with this intuition, i.e., yield high retain rates.

List of the most indicative features include "wonder(ful)", "best", "love", "trash", "great", "classic", "recommend", "terribl(e)", "perfect", "uniqu(e)", ""fail", "amaz(ing)", "fine", "supris(e)", "worst", "silli(y)", "flawless", "wast(e)", "dull" and "ridicul(ous)".

As discussed above, nouns or verbs are more often used to describe the movie plot, hence are less indicative, i.e., with smaller retain rates.

Some of the word features with low retaining probability-hence, possibly less indicative-include "year", "young", "possibl(e)", "happen", "dead", "music", "flick", "shot", "oscar", "kill", "spent", "pretti(y)", "say", "review", "support", "anim(ation)", "actual", "call", "cut", and "role".

One interesting observation is that we find the word "oscar" is also in the list of less informative features, which implies movie reviews and Academy Awards are not necessarily correlated.

In addition, we also include a list of popular and possibly unique named entities that are relevant to movie industry, including "baldwin", "niro", "spacey", "depp", "downey", "pitt", "pacino", "marilyn", "hepburn", "craig", "dench", "sammo", "clooney", "kidman", "mccarthi", "kermit", "godzilla", "nimoy", "shawshank", "yokai", "emraan", "kurosawa", "spielberg", "cameron", "pacino", "jackson", "eastwood", "allen", and "verhoeven".

We also notice interesting pattern on this list.

For example, some actors (e.g., "baldwin" and "kidman", etc.) and directors (e.g., "kurosawa" and "eastwood" etc.), yield high retain rates shortly after initial optimization.

The retain rates of actors like "downey" or "spacey", and directors like "spielberg" or "cameron" slightly increase or remain similar over time .

The word "pitt", 2 however, yields a declined retain rates throughout training, which suggests a less indicative feature for review classification.

Note that higher retain rate means the corresponding features are more indicative in classifying IMDB reviews into positive or negative labels, i.e., no explicit association with the label itself.

Imposing regularizaiton for a better model generalization is not a new topic.

However we tackle the problem for the dropout neural network regularization in a different way.

The theoretical upper bound we proved on the Rademacher complexity facilitates us to directly incorporate the dropout rates into the objective function.

In this way the dropout rate can be optimized by block coordinate descent procedure with one consistent objective.

Our empirical evaluation demonstrates promising results and interesting patterns on adapted retain rates.

In the future, we would like to investigate the sparsity property of the learnt retain rates to encourage a sparse representation of the data and the neural network structure BID26 , similar to the sparse Bayesian models and relevance vector machine BID21 .

We would also like to explore the applications of deep network compression (Han et al., 2015a; BID5 BID13 BID9 .

In addition, one other possible research direction is to dynamically adjust the architecture of the deep neural networks BID17 BID3 Guo et al., 2016) , and hence reduce the model complexity via dropout rates.

Features include words associated with the 20 largest and smallest retain rates, as well as a collection of movie related entities, e.g., actor, director, etc.

From our model, words like "love", "great", "terribl(e)", "perfect", "uniqu(e)", "fine(st)", "supris(e)", and "silli(y)" yield large retain rates and hence are indicative feature for predication (in the top half).

On the other hand, words like "say", "pretti(y)", "young", "review", "role", "anim(ation)" and "actual" have near zero retain rates upon model convergence, which are less informative.

For named entities that are relevant to movie industry, we also observe some interesting pattern.

Some actors (e.g., "baldwin" and "kidman", etc.) and directors (e.g., "kurosawa" and "eastwood" etc.), yield high retain rates shortly after initialization.

The retain rates of actors like "downey" or "spacey", and directors like "spielberg" or "cameron" slightly increase or remain similar throughout optimization.

Note that higher retain rate means the corresponding features are more indicative in classifying IMDB reviews into positive or negative labels, i.e., no explicit association with the label itself.

Proof.

In the analysis of Rademacher complexity, we treat the functions fed into the neurons of the l th layer as one function class F l = f l (x; w :l ).

Here again we are using the notation w :l = {w 1 , . . .

, w l }, and w = w :L .

As a consequence ???j, f DISPLAYFORM0 Note here f L (x; W ) used in section 3 is a vector, but f L (x; w) used in this subsection is a scalar.

The connection between f L (x; W ) and f L (x; w) is that each dimension of f L (x; W ) is viewed as one instance coming from the same function classs f L (x; w).

Similar ways of proof have been adopted in BID24 .To simplify our analysis, we follow BID24 and reformulate the cross-entropy loss on top of the softmax into a single logistic function DISPLAYFORM1 .The function class fed into the neurons of the l th layer f l (x; w :l ) admits a recursive expression DISPLAYFORM2 Given the neural network function (1) and the logistic loss function l is 1 Lipschitz, by Contraction lemma (a variant of the lemma 26.9 on page 381, Chapter 26 of BID16 ), the empirical Rademacher complexity of the loss function is bounded by DISPLAYFORM3 Note the empirical Rademacher complexity of the function class of the L th layer, i.e., the last output layer, is DISPLAYFORM4 To prove the bound in a recursive way, let's also define a variant of the Rademacher complexity with absolute value inside the supremum: DISPLAYFORM5 Note hereR S (f ) is not exactly the same as the Rademacher complexity defined before in this paper.

And we have DISPLAYFORM6 Now we start the recursive proof.

The empirical Rademacher complexity (with absolute value inside supremum) of the function class of the l th layer is DISPLAYFORM7 DISPLAYFORM8 By the calculous of Rademacher complexity, DISPLAYFORM9 Now we hav?? DISPLAYFORM10 Let g DISPLAYFORM11 Combining the inequalities (6), FORMULA22 , FORMULA5 , FORMULA5 , and (19), we have DISPLAYFORM12

Here we need to define truncated cross entropy loss function: DISPLAYFORM0 where C l is a constant.

Note with the truncation, the cross entropy loss is still 1-Lipschitz so the empirical Rademacher complexity bound still holds for the truncated lossl(f L (x; W, ??), y).Theorem 6.1.

For the dropout neural network defined in section (3), if truncated cross entropy lossl (21) is used, then ????? ??? 0, with probability at least 1 ??? 2??, ???l ??? f L : DISPLAYFORM1 Note here the empirical Rademacher complexity for the bounded loss function R S (l ??? f L ) admits the same bound as the empirical Rademacher complexity for the unbounded cross entropy loss R S (l ??? f L ).

In fact, adding a Rademacher related regularizer, though not investigated much, is not new at least for linear functions.

It is well known BID16 ) that the empirical Rademacher complexity of the linear class H 2 = {x ??? w, x : w 2 ??? B 2 } is bounded by R S ??? max i x i 2 B 2 / ??? n.

Note the l 2 loss function is 2-Lipschtz.

In this way, we may interpret the regularizer in the ridge regression related an upper bound for the empirical Rademacher complexity of the linear function class.

H 1 = {x ??? w, x : w 1 ??? B 1 }, the empirical Rademacher complexity is bounded by DISPLAYFORM0 So the lasso problem can also be viewed as adding a Rademacher-related regularization to the empirical loss minimization objective.

In application we do not want the regularization term to vary too much when the neural network has different number of internal neurons.

To overcome that we design some heuristics to add to the regularizer.

Note here all the scales mentioned in this section are added in a heuristic fashion.

It is purely empirical.

When p = q = 2, the regularizer is bounded by DISPLAYFORM0 Note that the content in Section 6.3 and 6.4 is based purely on heuristics, and derived on an ad hoc basis.

DISPLAYFORM1 Similarly, when p = ??? and q = 1, the scaled regularizer we used is DISPLAYFORM2

In this sub-section we demonstrate the dropout rates convergence acrross multiple runs.

We use a neural network with one hidden layer of 1024 ReLU units to illustrate, and feed it with MNIST dataset.

We train the network for 10 different runs, with same configurations and empirical settings, except different initializations on the network weight coefficients.

FIG5 shows the histogram of hidden layer retain rates upon model convergence under different runs.

We observe similar dropout behavior and distribution among multiple runs upon model convergence, i.e., the histograms of the retain rates do not diverge much across different runs in regards to different model weight initializations.

<|TLDR|>

@highlight

We propose a novel framework to adaptively adjust the dropout rates for the deep neural network based on a Rademacher complexity bound.

@highlight

The authors connect dropout parameters to a bound of the Rademacher complexity of the network

@highlight

Relates complexity of networks' learnability to dropout rates in backpropagation.