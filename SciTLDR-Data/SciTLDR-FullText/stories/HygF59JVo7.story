Classification systems typically act in isolation, meaning they are required to implicitly memorize the characteristics of all candidate classes in order to classify.

The cost of this is increased memory usage and poor sample efficiency.

We propose a model which instead verifies using reference images during the classification process, reducing the burden of memorization.

The model uses iterative non-differentiable queries in order to classify an image.

We demonstrate that such a model is feasible to train and can match baseline accuracy while being more parameter efficient.

However, we show that finding the correct balance between image recognition and verification is essential to pushing the model towards desired behavior, suggesting that a pipeline of recognition followed by verification is a more promising approach towards designing more powerful networks with simpler architectures.

FIG3 : Overview of our hybrid model in contrast with two opposing approaches.

(a) Recognition network directly predicts the class given the input.

(b) Verification network predicts binary output indicating the amount of similarity or likelihood that they belong in the same class.

The verification network can be used to compare to all reference images from each class to produce the final class prediction.

(c) Our approach, RVNN, queries for reference image from a particular class at each time step, and makes a class prediction at the last time step.

BID9 .

Then to classify, the image is pair-wise compared with a support image from every class and 37 image with the maximum similarity score is chosen.

Matching networks extends the verification-38 based approach by outputting a prediction based on a weighted-sum of similarity across classes [18] .

Additionally the work introduces an episodic-training regime which encourages the model to better 40 learn for the one-shot learning scenario.

Prototypical Networks uses Euclidean Distance in embedding 41 space as a verification metric rather than a learned metric, while maintaining the same training regime 42 as Matching Networks to encourage different classes to have distant means in embedding space [16] .

One recent work outside of few-shot learning domain is the Retrieval-Augmented Convolutional

Neural Networks (RaCNN) [21] , which combines CNN recognition network with a retrieval engine 45 for support images to help increase adversarial robustness.

For all the above few shot learning approaches, verification with support images from all classes are required before a classification decision is made.

Hence the classification decision is solely derived 48 from verifications.

RaCNN is closer to our approach, which uses a hybrid between verification 49 and recognition.

However, RaCNN simply retrieves the K closest support image neighbours in the 50 embedding space, whereas our model is required to form a hypothesis as to which class to compare 51 with.

In cases in which there are a large number of classes, we expect our approach to excel.

As well,

this introduces a non-differentiable component in our model not present in previous work.

Prior work has also looked at the concept of incorporating external knowledge into the decision 55 making process of neural networks, often with non-differentiable components.

Buck order to produce a prediction C for the class.

FIG2 illustrates the full process.

The subsequent sections detail the implementations of the three components as well as training 84 considerations.

The recurrent querying model f rnn was implemented using a Gated Recurrent Unit (GRU) [3] .

We 105 also considered passing in additional information to f rnn , such as the query that was used in the 106 previous time steps.

We implemented f q as sampling a class based on the categorical probability given by the softmax of 109 the logits from the f rnn .

Therefore, f q can be written as: DISPLAYFORM0 DISPLAYFORM1 To sample S n+1 , we can use the Gumbel-Max trick [5, 12] : DISPLAYFORM2 where g i ...g k are i.i.d samples drawn from Gumbel(0, 1) distribution.

However, the arg max operator 112 is not differentiable, so instead we explored two approaches during training.

The first is to use the Gumbel-Softmax trick, also known as the Concrete estimator [7, 11] .

We 114 relaxed the arg max operator to a differentiable softmax function with temperature parameter ?? : DISPLAYFORM0 The ?? parameter is annealed exponentially from ?? = 1 to ?? = 0.5 as the training iterations progresses.

The second approach is to use simple Straight-Through estimator BID0 .

In the forward pass, we apply 117 the Gumbel-Max trick to take discrete query choices.

Then on the backward pass, we set the derivative 118 of the query with respect to the softmax probabilities to be identity so that the out-going gradient 119 from the arg max operator is equal to the incoming gradient during backpropagation: DISPLAYFORM0

We highlight that our model differ from several existing networks in various aspects.

The performance of the model is assessed by both reduced parameter usage and sample efficiency.

Reduced parameter usage is measured relative to a baseline model, in this case the CNN architecture

The performance of the model alone does not indicate whether our approach is functioning as intended.

We also experiment with several small modifications to our architecture as well as a few hyper-

parameters that are unique to our model.

We list them here below.

??? Architectural Considerations

-Query Memory (QM): The query from the past time step is passed to the RNN.

the RNN learns to adapt its policy to varying levels of accuracy, and see whether a better result can 184 be achieved when they are used in tandem.

We also experiment with fixing query policy to assess 185 whether the RNN actually learns intelligent query behavior.

From our overall performance metrics we observe that at both smaller and larger sizes of model, our usage.

This is in agreement with our hypotheses that our model would be more parameter efficient.

RNN's learned policy is compared against a random query policy and the optimal query policy (never 218 repeating a query).

From the figure we see that the model is able to conduct a better than random 219 query policy, but is not able to achieve optimal performance.

We also observed that performance 220 increases with a higher RNN size up to 200.

This suggests that the model is in some part able to track 221 previously unsuccessful queries and remember if there was a match.

However, its memory is not 222 perfect and it cannot achieve optimal performance.

and a real (neural network) comparator.

Of the query-based models, informed queries performed the best over random and no query models.

appropriate pipeline for our model is to perform a recognition operation which is then followed by 241 verification, rather than perform them simultaneously.

This new pipeline would also imply that our model may be less well suited for the one-shot learning 243 task than initially believed, as a reasonably-well trained recognition module is required as the first tasks of this form are pre-requisite if neural-network models are considered to be as such BID5 .

@highlight

Image classification via iteratively querying for reference image from a candidate class with a RNN and use CNN to compare to the input image