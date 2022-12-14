This paper proposes a new model for the rating prediction task in recommender systems which significantly outperforms previous state-of-the art models on a time-split Netflix data set.

Our model is based on deep autoencoder with 6 layers and is trained end-to-end without any layer-wise pre-training.

We empirically demonstrate that: a) deep autoencoder models generalize much better than the shallow ones, b) non-linear activation functions with negative parts are crucial for training deep models, and c) heavy use of regularization techniques such as dropout is necessary to prevent over-fitting.

We also propose a new training algorithm based on iterative output re-feeding to overcome natural sparseness of collaborate filtering.

The new algorithm significantly speeds up training and improves model performance.

Our code is publicly available.

Sites like Amazon, Netflix and Spotify use recommender systems to suggest items to users.

Recommender systems can be divided into two categories: context-based and personalized recommendations.

Context based recommendations take into account contextual factors such as location, date and time BID0 .

Personalized recommendations typically suggest items to users using the collaborative filtering (CF) approach.

In this approach the user's interests are predicted based on the analysis of tastes and preference of other users in the system and implicitly inferring "similarity" between them.

The underlying assumption is that two people who have similar tastes, have a higher likelihood of having the same opinion on an item than two randomly chosen people.

In designing recommender systems, the goal is to improve the accuracy of predictions.

The Netflix Prize contest provides the most famous example of this problem BID1 : Netflix held the Netflix Prize to substantially improve the accuracy of the algorithm to predict user ratings for films.

This is a classic CF problem: Infer the missing entries in an mxn matrix, R, whose (i, j) entry describes the ratings given by the ith user to the jth item.

The performance is then measured using Root Mean Squared Error (RMSE).Training very deep autoencoders is non trivial both from optimization and regularization points of view.

Early works on training auto-enocoders adapted layer-wise pre-training to solve optimization issues BID5 .

In this work, we empirically show that optimization difficulties of training deep autoencoders can be solved by using scaled exponential linear units (SELUs) BID9 .

This enables training without any layer-wise pre-training or residual connections.

Since publicly available data sets for CF are relatively small, sufficiently large models can easily overfit.

To prevent overfitting we employ heavy dropout with drop probability as high as 0.8.

We also introduce a new output re-feeding training algorithm which helps to bypass the natural sparseness of updates in collaborative filtering and helps to further improve the model performance.

Deep learning BID13 has led to breakthroughs in image recognition, natural language understanding, and reinforcement learning.

Naturally, these successes fuel an interest for using deep learning in recommender systems.

First attempts at using deep learning for recommender systems involved restricted Boltzman machines (RBM) BID18 .

Several recent approaches use autoencoders BID19 BID20 , feed-forward neural networks BID4 , neural autoregressive architectures BID24 and recurrent recommender networks BID22 .

Many popular matrix factorization techniques can be thought of as a form of dimensionality reduction.

It is, therefore, natural to adapt deep autoencoders for this task as well.

I-AutoRec (item-based autoencoder) and U-AutoRec (user-based autoencoder) are first successful attempts to do so BID19 .

Stacked de-noising autoencoders has been sucesfully used on this task as well .There are many non deep learning types of approaches to collaborative filtering (CF) BID2 BID17 .

Matrix factorization techniques, such as alternating least squares (ALS) BID8 ) and probabilistic matrix factorization BID15 ) are particularly popular.

The most robust systems may incorporate several ideas together such as the winning solution to the Netflix Prize competition BID10 .

Note that Netflix Prize data also includes temporal signal -time when each rating has been made.

Thus, several classic CF approaches has been extended to incorporate temporal information such as TimeSVD++ BID11 , as well as more recent RNN-based techniques such as recurrent recommender networks BID22 .

Our model is inspired by U-AutoRec approach with several important distinctions.

We train much deeper models.

To enable this without any pre-training, we: a) use "scaled exponential linear units" (SELUs) BID9 , b) use high dropout rates, and d) use iterative output re-feeding during training.

An autoencoder is a network which implements two transformations -encoder encode(x) : DISPLAYFORM0 The "goal" of autoenoder is to obtain d dimensional representation of data such that an error measure between x and f (x) = decode(encode(x)) is minimized Hinton & Zemel (1994) .

Figure 1 depicts typical 4-layer autoencoder network.

If noise is added to the data during encoding step, the autoencoder is called de-noising.

Autoencoder is an excellent tool for dimensionality reduction and can be thought of as a strict generalization of principle component analysis (PCA) BID5 .

An autoencoder without non-linear activations and only with "code" layer should be able to learn PCA transformation in the encoder if trained to optimize mean squared error (MSE) loss.

In our model, both encoder and decoder parts of the autoencoder consist of feed-forward neural networks with classical fully connected layers computing l = f (W * x + b), where f is some nonlinear activation function.

If range of the activation function is smaller than that of data, the last layer of the decoder should be kept linear.

We found it to be very important for activation function f in hidden layers to contain non-zero negative part, and we use SELU units in most of our experiments (see Section 3.2 for details).If decoder mirrors encoder architecture (as it does in our model), then one can constrain decoder's weights W l d to be equal to transposed encoder weights W l e from the corresponding layer l. Such autoencoder is called constrained or tied and has almost two times less free parameters than unconstrained one.

Forward pass and inference.

During forward pass (and inference) the model takes user represented by his vector of ratings from the training set x ??? R n , where n is number of items.

Note that x is very sparse, while the output of the decoder, f (x) ??? R n is dense and contains rating predictions for all items in the corpus.

Since it doesn't make sense to predict zeros in user's representation vector x, we follow the approach from BID19 and optimize Masked Mean Squared Error loss: DISPLAYFORM0 where r i is actual rating, y i is reconstructed, or predicted rating, and m i is a mask function such that m i = 1 if r i = 0 else m i = 0.

Note that there is a straightforward relation between RMSE score and MMSE score: DISPLAYFORM1

During training and inference, an input x ??? R n is very sparse because no user can realistically rate but a tiny fractions of all items.

This poses problem for model training.

Bayesian approches can be used to overcome this issue .

On the other hand, autoencoder's output f (x) is dense.

Lets consider an idealized scenario with a perfect f .

Then f (x) i = x i , ???i : x i = 0 and f (x) i accurately predicts all user's future ratings for items i : x i = 0.

This means that if user rates new item k (thereby creating a new vector x ) then f (x) k = x k and f (x) = f (x ).

Hence, in this idealized scenario, y = f (x) should be a fixed point of a well trained autoencoder: f (y) = y.

To explicitly enforce fixed-point constraint and to be able to perform dense training updates, we augment every optimization iteration with an iterative dense re-feeding steps (3 and 4 below) as follows:1.

Given sparse x, compute dense f (x) and loss using equation 1 (forward pass) 2.

Compute gradients and perform weight update (backward pass) 3.

Treat f (x) as a new example and compute f (f (x)).

Now both f (x) and f (f (x)) are dense and the loss from equation 1 has all m as non-zeros. (second forward pass) 4.

Compute gradients and perform weight update (second backward pass)Steps (3) and (4) can be also performed more than once for every iteration.

For the rating prediction task, it is often most relevant to predict future ratings given the past ones instead of predicting ratings missing at random.

For evaluation purposes we followed BID22 exactly by splitting the original Netflix Prize BID1 training set into several training and testing intervals based on time.

Training interval contains ratings which came in earlier than the ones from testing interval.

Testing interval is then randomly split into Test and Validation subsets so that each rating from testing interval has a 50% chance of appearing in either subset.

Users and items that do not appear in the training set are removed from both test and validation subsets.

TAB0 provides details on the data sets.

For most of our experiments we uses a batch size of 128, trained using SGD with momentum of 0.9 and learning rate of 0.001.

We used xavier initialization to initialize parameters.

Note, that unlike BID20 we did not use any layer-wise pre-training.

We believe that we were able to do so successfully because of choosing the right activation function (see Section 3.2).

To explore the effects of using different activation functions, we tested some of the most popular choices in deep learning : sigmoid, "rectified linear units" (RELU), max(relu(x), 6) or RELU6, Figure 1 : AutoEncoder consists of two neural networks, encoder and decoder, fused together on the "representation" layer z. Encoder has 2 layers e 1 and e 2 and decoder has 2 layers d 1 and d 2 .

Dropout may be applied to coding layer z. hyperbolic tangent (TANH), "exponential linear units" (ELU) BID3 , leaky relu (LRELU) BID23 , "self-gated activation function" (SWISH) BID16 , and "scaled exponential linear units" BID9 ) (SELU) on the 4 layer autoencoder with 128 units in each hidden layer.

Because ratings are on the scale from 1 to 5, we keep last layer of the decoder linear for sigmoid and tanh-based models.

In all other models activation function is applied in all layers.

We found that on this task ELU, SELU and LRELU perform much better than SIGMOID, RELU, RELU6, TANH and SWISH.

FIG1 clearly demonstrates this.

There are two properties which seems to separate activations which perform well from those which do not: a) non-zero negative part and b) unbounded positive part.

Hence, we conclude, that in this setting these properties are important for successful training.

Thus, we use SELU activation units and tune SELU-based networks for performance.

The largest data set we use for training, "Netflix Full" from TAB0 , contains 98M ratings given by 477K users.

Number of movies (e.g. items) in this set is n = 17, 768.

Therefore, the first layer of encoder will have d * n + d weights, where d is number of units in the layer.

For modern deep learning algorithms and hardware this is relatively small task.

If we start with single layer encoders and decoders we can quickly overfit to the training data even for d as small as 512.

FIG2 clearly demonstrates this.

Switching from unconstrained autoencoder to constrained reduces over-fitting, but does not completely solve the problem.

While making layers wider helps bring training loss down, adding more layers is often correlated with a network's ability to generalize.

In this set of experiments we show that this is indeed the case here.

We choose small enough dimensionality (d = 128) for all hidden layers to easily avoid over-fitting and start adding more layers.

TAB1 shows that there is a positive correlation between the number of layers and the evaluation accuracy.

Going from one layer in encoder and decoder to three layers in both provides good improvement in evaluation RMSE (from 1.146 to 0.9378).

After that, blindly adding more layers does help, however it provides diminishing returns.

Note that the model with single d = 256 layer in encoder and decoder has 9,115,240 parameters which is almost two times more than any of these deep models while having much worse evauation RMSE (above 1.0).

Section 3.4 shows us that adding too many small layers eventually hits diminishing returns.

Thus, we start experimenting with model architecture and hyper-parameters more broadly.

Our most promising model has the following architecture: n, 512, 512, 1024, 512, 512, n, which means 3 layers in encoder (512,512,1024), coding layer of 1024 and 3 layers in decoder of size 512,512,n.

This model, however, quickly over-fits if trained with no regularization.

To regularize it, we tried several dropout values and, interestingly, very high values of drop probability (e.g. 0.8) turned out to be the best.

See FIG3 for evaluation RMSE.

We apply dropout on the encoder output only, e.g. f (x) = decode(dropout(encode(x))).

We tried applying dropout after every layer of the model but that stifled training convergence and did not improve generalization.

Iterative dense re-feeding (see Section 2.2) provides us with additional improvement in evaluation accuracy for our 6-layer-model: n, 512, 512, 1024, dp(0.8), 512, 512, n (referred to as Baseline below).

Here each parameter denotes the number of inputs, hidden units, or outputs and dp(0.8) is a dropout layer with a drop probability of 0.8.

Just applying output re-feeding did not have significant Figure 5 for details.

Figure 5: Effects of dense re-feeding.

Y-axis: evaluation RMSE, X-axis: epoch number.

Baseline model was trained with learning rate of 0.001.

Applying re-feeding step with the same learning rate almost did not help (Baseline RF).

Learning rate of 0.005 (Baseline LR 0.005) is too big for baseline model without re-feeding.

However, increasing both learning rate and applying re-feeding step clearly helps (Baseline LR 0.005 RF).Applying dense re-feeding and increasing the learning rate, allowed us to further improve the evaluation RMSE from 0.9167 to 0.9100.

Picking a checkpoint with best evaluation RMSE and computing test RMSE gives as 0.9099, which we believe is significantly better than other methods.

We compare our best model with Recurrent Recommender Network from BID22 which has been shown to outperform PMF BID15 , T-SVD BID11 and I/U-AR BID19 on the data we use (see TAB0 for data description).

Note, that unlike T-SVD and RRN, our method does not explicitly take into account temporal dynamics of ratings.

Yet, TAB2 shows that it is still capable of outperforming these methods on future rating prediction task.

We train each model using only the training set and compute evaluation RMSE for 100 epochs.

Then the checkpoint with the highest evaluation RMSE is tested on the test set."Netflix 3 months" has 7 times less training data compared to "Netflix full", it is therefore, not surprising that the model's performance is significantly worse if trained on this data alone (0.9373 vs 0.9099).

In fact, the model that performs best on "Netflix full" over-fits on this set, and we had to reduce the model's complexity accordingly (see TAB3 for details).

Netflix 3 months 0.9373 n, 128, 256, 256, dp(0.65), 256, 128, n Netflix 6 months 0.9207 n, 256, 256, 512, dp(0.8), 256, 256, n Netflix 1 year 0.9225 n, 256, 256, 512, dp(0.8), 256, 256, n Netfix Full 0.9099 n, 512, 512, 1024, dp(0.8), 512, 512, n

Deep learning has revolutionized many areas of machine learning, and it is poised do so with recommender systems as well.

In this paper we demonstrated how very deep autoencoders can be successfully trained even on relatively small amounts of data by using both well established (dropout) and relatively recent ("scaled exponential linear units") deep learning techniques.

Further, we introduced iterative output re-feeding -a technique which allowed us to perform dense updates in collaborative filtering, increase learning rate and further improve generalization performance of our model.

On the task of future rating prediction, our model outperforms other approaches even without using additional temporal signals.

While our code supports item-based model (such as I-AutoRec) we argue that this approach is less practical than user-based model (U-AutoRec).

This is because in real-world recommender systems, there are usually much more users then items.

Finally, when building personalized recommender system and faced with scaling problems, it can be acceptable to sample items but not users.

<|TLDR|>

@highlight

This paper demonstrates how to train deep autoencoders end-to-end to achieve SoA results on time-split Netflix data set.

@highlight

This paper presents a deep autoencoder model for rating prediction that outperforms other state-of-the-art approahces on the Netflix prize dataset. 

@highlight

Proposes to use a deep AE to do rating prediction tasks in recommender systems.

@highlight

The authors present a model for more accurate Netflix recommendations demonstrating that a deep autoencoder can out-perform more complex RNN-based models that have temporal information. 