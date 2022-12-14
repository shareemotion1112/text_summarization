One of the big challenges in machine learning applications is that training data can be different from the real-world data faced by the algorithm.

In language modeling, users’ language (e.g. in private messaging) could change in a year and be completely different from what we observe in publicly available data.

At the same time, public data can be used for obtaining general knowledge (i.e. general model of English).

We study approaches to distributed fine-tuning of a general model on user private data with the additional requirements of maintaining the quality on the general data and minimization of communication costs.

We propose a novel technique that significantly improves prediction quality on users’ language compared to a general model and outperforms gradient compression methods in terms of communication efficiency.

The proposed procedure is fast and leads to an almost 70% perplexity reduction and 8.7 percentage point improvement in keystroke saving rate on informal English texts.

Finally, we propose an experimental framework for evaluating differential privacy of distributed training of language models and show that our approach has good privacy guarantees.

Two common problems arising after deployment of a machine learning model on user devices are discrepancy between training data and actual data stored on user devices, and the need of regular model updates.

In the case of language modeling, it corresponds to the difference between language and style of the training corpus mined in the Internet and messages of the user, which account for most of the text generated on the device.

Even if the training corpus includes a substantial part of informal texts (tweets, forum threads, etc.), real user data can be very different.

This is a challenge for word prediction algorithms in software keyboard applications.

The most general approach to improvement of customer experience in typing is integrating a separate user language model trained on device in an on-line fashion.

In the simplest case it is a smoothed n-gram (e.g. Kneser-Ney n-gram model BID6 )).In BID26 continuously learned personalized language model based on LSTM was proposed but as far as each user generates only a small portion of textual data, such data by itself cannot be used for updates of the general model.

Thus, for a model update, a collection of potentially sensitive data from many users is needed.

As shown in , collecting data for training may be avoided.

We propose a similar approach for distributed fine-tuning of language models on private data.

In this sense our method can be considered as "federated fine-tuning" but we prefer to take more traditional term.

In this setting we start with a language model trained on a large text corpus representing the general language.

This model G will be updated continuously on user devices but with an additional requirement that the model must not go too far from the general language model, i.e. we don't overfit on user data.

We pursue two goals: 1) to develop an algorithm of distributed fine-tuning that is fast, communication efficient and doesn't need collecting sensitive user data; and 2) to prevent the language model from forgetting "general English".

Besides, we provide analysis of possibility of privacy violation After each round the server model G t+1 is sent to the next K elements.

in our model.

BID8 ) demonstrated an attack on distributed training algorithm leading to information leakage.

This means that privacy analysis in necessary for such algorithms.

Our main contributions are: 1) we propose an efficient procedure of distributed fine-tuning of language models immune to the problem of catastrophic forgetting BID3 ), 2) we provide experimental evaluation of on-device training time, communication costs and convergence rates of the general language model in realistic conditions, 3) we compare two most popular strategies of improving communication efficiency in the context of distributed learning, and 4) we propose an experimental framework for evaluation of differential privacy of distributed training of language models, and using this framework, we evaluate privacy guarantees of our approach.

In our research we are focused on improvement of keystroke saving rate (see section 2.4) because this metric reflects customer typing experience more directly than perplexity or BLEU.

We use LSTM architecture for our language model as described in BID27 and evaluate ondevice training time for this architecture.

We show that the on-device training time is reasonably small, thus demonstrating the feasibility of the whole approach.

As usual, our task is to predict the next word w N given a sequence of words w 1 . . .

w N −1 .

If the prediction algorithm of a software keyboard application is based on a language model with low perplexity on the test data, the application provides a reasonably sorted list of input candidates.

Of course, the test data should be drawn from the same distribution as the user data.

In our case we also want to have only one, continuously improving model on a user device.

As far as the user can always switch to the general English, we have to prevent the model from overfitting on the texts written on the device, or catastrophic forgetting BID16 ; BID5 ; BID10 ).Our approach can be summarized as follows FIG0 ): 0) At the first stage we have an initial language model G 0 (at every step t it will be updated to G t ) trained on a large corpus of standard English; 1) As soon as a user inputs sufficient volume of text, the latest version of G t is sent from the server to provide updates, and fine-tuning starts on the device leading to the modelḠ i t ; 2) When the training is finished the modelḠ i t is sent back to the server; 3) Every time the updated modelsḠ i t are received from K different users, a round of model update is run resulting in the model G t+1 2.1 LEARNING WITHOUT FORGETTING In its original formulation BID13 ), the problem of learning without forgetting (LwF) consists in re-training of existing model Θ on new data such that its performance on the old data does not degrade.

More formally, suppose we have a classifier with a set of parameters Θ trained and tested on a dataset D = {Tr, Ts} where Tr and Ts are train and test sets accordingly.

Let D * = {Tr * , Ts * } be some new dataset.

Our goal is to update the parameters Θ with dataset D = {Tr * , Ts ∪ Ts * }i.e. we have to provide the best performance on old and new types of data having only training data of the new type.

In contrast, joint training BID1 ) assumes a model update with access to the both datasets: DISPLAYFORM0 As we want to avoid sending user data to the server, classical joint training is impossible.

On the other hand, LwF seems promising.

In this case we send the user a current instance of the general language model G t with weights θ g and fine-tune it producing the model θ u , while θ g is used for generating predictions for regularization.

The resulting loss at step t and true word w t can be calculated as follows: DISPLAYFORM1 where DISPLAYFORM2 A similar approach is taken in BID23 where predictions of a basic model (in this case θ g ) are taken as soft labels.

Minimizing loss in FORMULA1 - FORMULA2 is equivalent to minimizing Kullback-Leibler divergence L(θ u ) = KL (P gr P u ) with respect to parameters θ u of P u where density of P gr is given by: DISPLAYFORM0 In (3) P T r * (x) stands for the real distribution on a user device and P (x|θ g ) is a probability given by the model of "general English" θ g .

It suggests that instead of optimizing L(θ u ) we can simply add data from Tr to Tr * to obtain the (1 − λ) portion.

This approach, called random rehearsal, was presented in BID22 .In practice in the case of fine-tuning with rehearsal a portion of the general English training corpus (standard English corpus) must be sent to the user device.

Volume of typical user data generated on device is of the order of tens of kilobytes per month, and the size of the training data sent to the device will be of the same order.

Overall, random rehearsal is more efficient, because there is no need to calculate soft labels.

The server-side part of the solution must aggregate modelsḠ i t from many users and use them to update the general model G t .

We took simple model averaging as a baseline solution and transfer learning BID0 BID24 ) as an alternative approach.

In the case of transfer learning we optimized cross-entropy function (1), with y * i given by an average prediction from N aggregated models θ k u : DISPLAYFORM0 Just as in the case of on-device training, transfer learning-based approach is rather inefficient in terms of time and memory because predictions from all models are needed.

Keystroke saving rate (KSS) BID17 ) is defined as a relative decrease in the number of characters the user has to type, given suggestions from the software keyboard: where N total is the total number of non-space characters in the typed text and N typed is the number of characters user still had to type until the correct suggestion was presented.

In our experiments we used top-3 suggestion lists.

DISPLAYFORM0 From the definition above one can see that KSS is better for customer experience assessment compared to perplexity.

Besides, perplexity measure underestimates out-of-vocabulary (OOV) words.

In the presence of OOV words perplexity is ill-defined, so all OOV words must be removed from the test set.

It makes a direct comparison of models with different vocabularies impossible, which is impractical.

Finally, our experiments have demonstrated that a small decrease in perplexity may not correspond to KSS improvement and doesn't lead to any practical result.

Nevertheless, our method demonstrates considerable perplexity reduction as well.

The goal of our experiments was to find the most efficient pipeline to distributed fine-tuning of language models.

We compared several approaches for client-side and server-side model updates.

In accordance with the problem statement we assumed a substantial difference between the reallife user corpus and the standard English corpus used for initial training, so we took Twitter and Wikipedia corpora for the user and standard English corpora correspondingly.

The standard English train dataset contained approximately 30M tokens.

The hyperparameters of the model were initially tuned on the Standard English validation set of 3.8M tokens.

The user train dataset contained approximately 1.7M tokens.

Updated models were tested on subsets of the Twitter and Wikipedia corpora containing 200k and 170k tokens correspondingly.

Comparison between the random rehearsal and LwF training methods were carried out on a single node.

For our experiments we used LSTM architecture from BID27 with 2x650 LSTM layers, a vocabulary size of 30k, dropout 0.5, minibatch size 20, BPTT steps 35.

The initial general English model was trained in 39 epochs.

We report KSS and perplexity on both the standard English test set and the user data test sets.

In the case of the standard English test set KSS was calculated on a subset of 200 sentences (3600 tokens).

The initial general English model had a perplexity of 100.1 and 67.9% KSS rate on the Standard English test and perplexity 336.0 and 49.7% KSS rate on the user data test set.

So, the model experienced a considerable 18.2% drop in performance on the user data test set.

TAB0 summarizes our experiments with on-device model update algorithms.

We see that the performance gap between the standard English and the user test sets can be considerably reduced at the cost of performance degradation on the first dataset.

The best average perplexity is reached with the random rehearsal method and λ = 0.5.

We believe that the reason of the comparably inferior performance of the LwF method can be explained by the fact that soft labels used by LwF give a poor approximation of the true word distribution of general English so adding a small portion of true data gives better results in terms of knowledge preservation.

To compare model averaging and transfer learning for a server-side model update, we carried out a small experiment with 10 nodes and 1 iteration of the server-side update.

Each model was trained on a mobile phone with a quad-core mobile CPU with a clock frequency 2.31 GHz.

We used a minibatch size 10, number of BPTT steps 20, learning rate 0.75 and 1 epoch.

Training took approximately 140 seconds on 20 kilobytes of text (user-generated and rehearsal data).

Note that we used mobile CPU only, so computation time may be reduced by using mobile GPU.

Due to absence of the frameworks that make backpropagation on a device possible we had to implement our own training on the phone.

After training the updated user models were used for general model update on the server.

For the server-side model update algorithm we also tried the approach proposed in BID23 .

In this case the new model is trained on the texts generated by its previous round of update.

We tested both 1 generation per epoch and a single time generation before the first epoch.

We carried out at most 6 epochs so we had 1 and 5 cycles of text generation correspondingly.

Results of the experiment are summarized in TAB1 .

We saw no significant differences between transfer learning on real and generated data.

The difference between transfer learning and averaging is more sound but still not large.

At the same time model averaging is much more computationally efficient, as long as transfer learning requires calculation of labels from each of the teacher models.

After 300 rounds of model updates with 3000 nodes (10 nodes per round) we ended up with an 8.7 absolute gain in KSS on the user data test with only a 0.6 absolute KSS drop on the standard English data test.

FIG1 shows that the model starts to perform reasonably well after 100 rounds of updates.

It also shows the importance of rehearsal for preventing catastrophic forgetting.

There are several strategies that help to make distributed learning communication efficient.

The most successful ones can be divided into two classes: 1) strategies that increase computation on nodes thus sending data to the server less frequently ), and 2) strategies that transmit only some part of data from devices to the server in a single round of averaging BID14 ; BID11 .

One of the most impressive results was reached by the Deep Gradient Compression BID14 ).

It belongs to the second class -its key idea is to send only the most important weight updates obtained during on-device training while accumulating the remaining ones in order to send them when the sum becomes large enough.

It was shown that Deep Gradient Compression method (DGC) allows to send a very small part of weight updates (0.1%) in a single round of averaging without loss in the quality of the model.

For language modeling task, the gradient compression ratio of 462x was obtained using gradient accumulation strategy for small updates.

However, DGC supposes that in each round of averaging only one user's model update is made for every node while methods from the first class increase computation on nodes to several epochs before model averaging.

In our experiments (2.5) we chose to train models on devices for one epoch rather than using DGC-style strategy.

As shown in TAB2 , this results in a total amount of 1.7Gb of data transmitted from nodes to the server in a single round (this amount certainly depends linearly on the size of the model).

We used a classical 2-layer LSTM model from BID27 but there are models that perform similarly or better but have less parameters (e.g. BID9 , BID21 ), so in practice we can improve the results shown in TAB2 .To prove competitiveness of our approach, we made the experiment (see TAB3 ) in the settings presented in BID14 .

We compared two strategies for improving communication efficiency: increasing computation on nodes and DGC.

The models were trained on a popular language modeling benchmark PTB.

The neural network architecture (2-layer LSTM with 1500 units, tied input and output embeddings and variational dropout with probability 0.65) as well as the results for DGC were taken from BID14 .

As for the first strategy, we trained the model for 28 rounds.

During the first round, a randomly initialized model was trained on the first node, then sent to the second node, trained there, and so on.

When training on the last (fourth) node was finished, the updated model was sent to all four nodes and the second round started.

The remaining 27 rounds were standard rounds of model averaging.

We had to make the first round so specific because we needed to simulate some kind of "pretraining" (which the task itself didn't suggest) in order to make model averaging perform well.

Since we had only one training corpus, no rehearsal was applied during training on nodes.

The number of training epochs on a node and learning rate decreased from 10-20 and 1.0 correspondingly in the first rounds to 1-3 and 0.27 in the last ones.

We used minibatch size 20 and 35 BPTT steps.

The first strategy achieved better perplexity with the same amount of data sent from nodes to the server compared to DGC.

The important thing is that the number of communications for it was 112 which is much less than 53k for DGC.

Since communication efficiency involves not only the data that is transmitted from devices to the server but also the time that is necessary to set up connections, we can conclude that increasing computation on nodes perfroms better in terms of communication efficiency than gradient compression methods.

This is why we chose the first strategy in our approach.

Moreover, in our scheme the data on a device is used only once and can be deleted after the on-device training whereas in DGC and many other distributed learning schemes the data on each device is used many times (once per epoch).Certainly, the two classes of strategies for improving communication efficiency are not mutually exclusive -we can apply DGC or, for example, methods that are described in BID11 to further reduce communication costs but this is out of the scope of the present paper.3 PRIVACY ANALYSIS

Our analysis is based on the experimental evaluation of differential privacy.

The notion of differential privacy BID2 ) appears naturally in many applications when it comes to estimating of the possibility of privacy violation.

In particular, it can be applied to language models trained on private user data.

Loosely speaking, if we have a mechanism that takes some input data and produces some output then differential privacy measures how a single input unit influences the total output.

In order to achieve differential privacy, some randomness must be introduced into the mechanism.

Definition 1.

A randomized mechanism M with domain D and range S satisfies (ε, δ)-differential privacy if for any two inputs d, d ∈ D that are adjacent (i.e. differ in one record) and for any subset of outputs S ⊆ S it holds that: DISPLAYFORM0 In our case D is the set of all subsets of users and a randomized mechanism M(d) is a mechanism that generates texts according to a certain language model trained on d ∈ D. Note that for any d we need to have DISPLAYFORM1 Thus it is necessary for S to be the set of all possible texts of some fixed length rather than the set of all texts of an arbitrary length.

In our analysis we will consider only the space of texts containing 10 words.

This is reasonable because it is close to the average length of a sentence in our user data corpus and it seems that if user's privacy is violated then 10 consequent words are already enough for an adversary to retrieve important information.

Let us fix two adjacent sets of users d and d , train models θ and θ on them and introduce random variable c(s).

It is defined by the expression DISPLAYFORM2 for any s ∈ S. Since a language model Θ assigns some positive probability to any sequence of words, c(s) is defined correctly for all s ∈ S.Parameter δ in the Definition 1 stands for the probability that two probabilities P (s|θ) and P (s|θ ) differ much.

This fact is formalized by the following proposition: DISPLAYFORM3 Proof.

Let B = {s ∈ S : c(s) > e ε }.

Then for any S ⊆ S DISPLAYFORM4 The proposition implies that it is sufficient to estimate the tail of the distribution of c(s) under measure P(·|θ).

Furthermore, FIG2 suggests that the tail of the empirical distribution function of the observed variable c(s) has the Pareto distribution.

This seems natural as far as words in human language follow Zipf's law which is a discrete analogue of the Pareto distribution.

To make a confident estimation of differential privacy parameters, we consider 20 different pairs of adjacent sets of users d and d .

For each one, we consider a composite null hypothesis that the tail of the random variable c(s) defined in (6) has the Pareto distribution with the shape parameter equal to its Hill's estimator (M. Hill (1975) ).

Then we apply the Lilliefors test and accept the null hypothesis at a significance level of 5%.

Quantiles of the Pareto distribution can be written down explicitly thus giving the following formula for estimation of parameters ε and δ: DISPLAYFORM5 where α and C are parameters of Pareto distribution defined in statistical tests (see Appendix).Finally, for a given δ we take the largest value of ε amongst all the experiments.

The critical value for the Lilliefors test at 5% significance level is 1.08.

In 19 cases out of 20 the Lilliefors test fails to reject the null hypothesis.

This conclusion, together with sample visual representation in FIG2 , allows us to state that the random variable c(s) indeed has tails that decrease like the Pareto distribution tails with quite a big shape parameter.

Exact values of KS statistics and Hill's estimators of this parameter for different pairs of users are provided in the TAB4 .

Table 6 shows the results for different values of δ calculated by formula (7).

In this table the value of ε is the largest value of this parameter in all 20 experiments.

The total number of users is 3 · 10 3 so it is reasonable to put δ = 10 −4 .

For this choice of δ parameter ε equals to 0.67.

It means that our algorithm offers reasonable privacy guarantees (see BID20 ).

Additionally we provide values of ε for smaller values of δ.

The results shown in Table 6 demonstrate that our scheme provides a very good level of privacy protection.

However, it is necessary to say that we only aim to produce an empirical estimation of differential privacy which inevitably holds with some high probability but not almost surely (this fact makes our approach close to the so-called random differential privacy introduced in BID7 ).

In many machine learning algorithms, the outcome is initially deterministic and some wellknown distribution is used to generate noise in order to make the algorithm differentially private (e.g. BID20 ).

In our mechanism the source of randomness lies inside the neural network and the output distributions can't be written explicitly.

This is the reason why we are able to provide only empirical estimations of differential privacy parameters.

We have presented our results in distributed fine-tuning of neural language models.

We paid special attention to preventing a catastrophic forgetting of the general language after a model fine-tuning on the user devices.

Our experiments showed that the performance of an initial model of the general English on user data can be improved significantly almost without a performance degradation on the standard English training data.

We found that a combination of on-device training with random rehearsal and server-side model averaging provides the best performance for such distributed finetuning.

Users' models were trained for the whole epoch that reduced communication costs while at the same time being quite fast -it took less than 3 minutes with a realistic assessment of volume of the available user data.

Finally, we provided an experimental evaluation of differential privacy of our method and showed that the method has a reasonable level of differential privacy compared to other solutions.

We still have to note that we provided an empirical estimation of differential privacy which holds with some high probability but not almost surely.

This statistic doesn't converge to the Kolmogorov distribution as shown in W. Lilliefors (1969) .

It converges to the distribution with smaller critical values at the same significance levels because we overfit on the sample data when the estimator r is plugged in.

We chose a 5% significance level and critical value for it is 1.08.

In 19 cases out of 20 the Lilliefors test failed to reject the null hypothesis at a 5% significance level.

TAB4 provides exact values obtained during the application of the statistical test.

Relying on these values along with data visualization in 3 we can state that random variable c(s) has tails that decrease like the Pareto distribution tails.

The hypothesis that we accepted suggests that the cumulative distribution function of c(s) is given by the formula (8).

It means that the tail distribution function for all x > x 0 is given by DISPLAYFORM0 We chose x 0 = c (k) n , so F (x 0 ) is just the ratio k/n.

Thus, C can be estimated by DISPLAYFORM1 Values of C are given in the TAB4 .

Finally, from formula (11) and proposition 1 it is easy to derive that (ε, δ)-differential privacy is provided by the values ε, δ that satisfy DISPLAYFORM2

<|TLDR|>

@highlight

We propose a method of distributed fine-tuning of language models on user devices without collection of private data

@highlight

This paper deals with improving language models on mobile equipments based on small portion of text that the user has inputted by employing a linearly interpolated objectives between user specific text and general English. 