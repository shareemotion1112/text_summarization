Federated learning is a recent advance in privacy protection.

In this context, a trusted curator aggregates parameters optimized in decentralized fashion by multiple clients.

The resulting model is then distributed back to all clients, ultimately converging to a joint representative model without explicitly having to share the data.

However, the protocol is vulnerable to differential attacks, which could originate from any party contributing during federated optimization.

In such an attack, a client's contribution during training and information about their data set is revealed through analyzing the distributed model.

We tackle this problem and propose an algorithm for client sided differential privacy preserving federated optimization.

The aim is to hide clients' contributions during training, balancing the trade-off between privacy loss and model performance.

Empirical studies suggest that given a sufficiently large number of participating clients, our proposed procedure can maintain client-level differential privacy at only a minor cost in model performance.

Lately, the topic of security in machine learning is enjoying increased interest.

This can be largely attributed to the success of big data in conjunction with deep learning and the urge for creating and processing ever larger data sets for data mining.

However, with the emergence of more and more machine learning services becoming part of our daily lives, making use of our data, special measures must be taken to protect privacy.

Unfortunately, anonymization alone often is not sufficient BID12 ; BID1 and standard machine learning approaches largely disregard privacy aspects and are susceptible to a variety of adversarial attacks BID11 .

In this regard, machine learning can be analyzed to recover private information about the participating user or employed data as well ; BID16 ; BID3 ; BID6 .

BID2 propose a measure for to assess the memorization of privacy related data.

All the aspects of privacy-preserving machine learning are aggravated when further restrictions apply such as a limited number of participating clients or restricted communication bandwidth such as mobile devices Google (2017) .In order to alleviate the need of explicitly sharing data for training machine learning models, decentralized approaches have been proposed, sometimes referred to as collaborative BID15 or federated learning BID9 .

In federated learning BID9 a model is learned by multiple clients in decentralized fashion.

Learning is shifted to the clients and only learned parameters are centralized by a trusted curator.

This curator then distributes an aggregated model back to the clients.

However, this alone is not sufficent to preserve privacy.

In BID14 it is shown that clients be identified in a federated learning setting by the model updates alone, necessitating further steps.

Clients not revealing their data is an advance in privacy protection.

However, when a model is learned in conventional way, its parameters reveal information about the data that was used during training.

In order to solve this issue, the concept of differential privacy (dp) BID4 for learning algorithms was proposed by BID0 .

The aim is to ensure a learned model does not reveal whether a certain data point was used during training.

We propose an algorithm that incorporates a dp-preserving mechanism into federated learning.

However, opposed to BID0 we do not aim at protecting w.r.t.

a single data point only.

Rather, we want to ensure that a learned model does not reveal whether a client participated during decentralized training.

This implies a client's whole data set is protected against differential attacks from other clients.

Our main contributions: First, we show that a client's participation can be hidden while model performance is kept high in federated learning.

We demonstrate that our proposed algorithm can achieve client level differential privacy at a minor loss in model performance.

An independent study BID10 , published at the same time, proposed a similar procedure for client level-dp.

Experimental setups however differ and BID10 also includes elementlevel privacy measures.

Second, we propose to dynamically adapt the dp-preserving mechanism during decentralized training.

Empirical studies suggest that model performance is increased that way.

This stands in contrast to latest advances in centralized training with differential privacy, were such adaptation was not beneficial.

We can link this discrepancy to the fact that, compared to centralized learning, gradients in federated learning exhibit different sensibilities to noise and batch size throughout the course of training.

In federated learning BID9 , communication between curator and clients might be limited (e.g. mobile phones) and/or vulnerable to interception.

The challenge of federated optimization is to learn a model with minimal information overhead between clients and curator.

In addition, clients' data might be non-IID, unbalanced and massively distributed.

The algorithm 'federated averaging' recently proposed by BID9

A lot of research has been conducted in protecting differential privacy on data level when a model is learned in a centralized manner.

This can be done by incorporating a dp-preserving randomized mechanism (e.g. the Gaussian mechanism) into the learning process.

We use the same definition for differential privacy in randomized mechanisms as BID0 :

A randomized mechanism M : D ??? R, with domain D and range R satisfies ( , ??)-differential privacy, if for any two adjacent inputs d, d ??? D and for any subset of outputs S ??? R it holds that DISPLAYFORM0 In this definition, ?? accounts for the probability that plain -differential privacy is broken.

The Gaussian mechanism (GM) approximates a real valued function f : D ??? R with a differentially private mechanism.

Specifically, a GM adds Gaussian noise calibrated to the functions data set sensitivity S f .

This sensitivity is defined as the maximum of the absolute distance DISPLAYFORM1 In the following we assume that ?? and are fixed and evaluate an inquiry to the GM about a single approximation of f (d).

We can then bound the probability that -dp is broken according to: ?? ??? 5 4 exp(???(?? ) 2 /2) (Theorem 3.22 in BID5 ).

It should be noted that ?? is accumulative and grows if the consecutive inquiries to the GM.

Therefore, to protect privacy, an accountant keeps track of ??.

Once a certain threshold for ?? is reached, the GM shall not answer any new inquires.

Recently, BID0 proposed a differentially private stochastic gradient descent algorithm (dp-SGD).

dp-SGD works similar to mini-batch gradient descent but the gradient averaging step is approximated by a GM.

In addition, the mini-batches are allocated through random sampling of the data.

For being fixed, a privacy accountant keeps track of ?? and stops training once a threshold is reached.

Intuitively, this means training is stopped once the probability that the learned model reveals whether a certain data point is part of the training set exceeds a certain threshold.

We propose to incorporate a randomized mechanism into federated learning.

However, opposed to Abadi et al. (2016) we do not aim at protecting a single data point's contribution in learning a model.

Instead, we aim at protecting a whole client's data set.

That is, we want to ensure that a learned model does not reveal whether a client participated during decentralized training while maintaining high model performance.

In the framework of federated optimization BID9 , the central curator averages client models (i.e. weight matrices) after each communication round.

In our proposed algorithm, we will alter and approximate this averaging with a randomized mechanism.

This is done to hide a single client's contribution within the aggregation and thus within the entire decentralized learning procedure.

The randomized mechanism we use to approximate the average consists of :??? Random sub-sampling (step 1 in FIG0 ): Let K be the total number of clients.

In each communication round a random subset Z t of size m t ??? K is sampled.

The curator then distributes the central model w t to only these clients.

The central model is optimized by the clients' on their data.

The clients in Z t now hold distinct local models {w k } mt k=0 .

The difference between the optimized local model and the central model will be referred to as client k's update ???w k = w k ??? w t .

The updates are sent back to the central curator at the end of each communication round.??? Distorting (step 3 and 4 in FIG0 ): A Gaussian mechanism is used to distort the sum of all updates.

This requires knowledge about the set's sensitivity with respect to the summing operation.

We can enforce a certain sensitivity by using scaled versions instead of the true updates: DISPLAYFORM0 ).

Scaling ensures that the second norm is limited ???k, w k 2 < S. The sensitivity of the scaled updates with respect to the summing operation is thus upper bounded by S. The GM now adds noise (scaled to sensitivity S) to the sum of all scaled updates.

Dividing the GM's output by m t yields an approximation to the true average of all client's updates, while preventing leakage of crucial information about an individual.

A new central model w t+1 is allocated by adding this approximation to the current central model w t .

DISPLAYFORM1 Gaussian mechanism approximating sum of updatesWhen factorizing 1/m t into the Gaussian mechanism, we notice that the average's distortion is governed by the noise variance S 2 ?? 2 /m.

However, this distortion should not exceed a certain limit.

Otherwise too much information from the sub-sampled average is destroyed by the added noise and there will not be any learning progress.

GM and random sub-sampling are both randomized mechanisms. (Indeed, BID0 used exactly this kind of average approximation in dp-SGD.

However, there it is used for gradient averaging, hiding a single data point's gradient at every iteration).

Thus, ?? and m also define the privacy loss incurred when the randomized mechanism provides an average approximation.

In order to keep track of this privacy loss, we make use of the moments accountant as proposed by Abadi et al. BID0 .

This accounting method provides much tighter bounds on the incurred privacy loss than the standard composition theorem (3.14 in BID5 ).

Each time the curator allocates a new model, the accountant evaluates ?? given , ?? and m. Training shall be Step 1: At each communication round t, m t out of total K clients are sampled uniformly at random.

The central model w t is distributed to the sampled clients.

Step 2: The selected clients optimize w t on their local data, leading to w k .

Clients centralize their local updates: DISPLAYFORM2 Step 3: The updates are clipped such that their sensitivity can be upper bounded.

The clipped updates are averaged.

Step 4: The central model is updated adding the averaged, clipped updates and distorting them with Gaussian noise tuned to the sensitivity's upper bound.

Having allocated the new central model, the procedure can be repeated.

However, before starting step 1, a privacy accountant evaluates the privacy loss that would arise through performing another communication round.

If that privacy loss is acceptable, a new round may start.

stopped once ?? reaches a certain threshold, i.e. the likelihood, that a clients contribution is revealed gets too high.

The choice of a threshold for ?? depends on the total amount of clients K. To ascertain that privacy for many is not preserved at the expense of revealing total information about a few, we have to ensure that ?? 1 K , refer to Dwork & Roth (2014) chapter 2.3 for more details.

Choosing S: When clipping the contributions, there is a trade-off.

On the one hand, S should be chosen small such that the noise variance stays small.

On the other hand, one wants to maintain as much of the original contributions as possible.

Following a procedure proposed by BID0 , in each communication round we calculate the median norm of all unclipped contributions and use this as the clipping bound S = median{ w k } k???Zt .

We do not use a randomized mechanism for computing the median, which, strictly speaking, is a violation of privacy.

However, the information leakage through the median is small (Future work will contain such a privacy measure).Choosing ?? and m: for fixed S, the ratio r = ?? 2 /m governs distortion and privacy loss.

It follows that the higher ?? and the lower m, the higher the privacy loss.

The privacy accountant tells us that for fixed r = ?? 2 /m, i.e. for the same level of distortion, privacy loss is smaller for ?? and m both being small.

An upper bound on the distortion rate r and a lower bound on the number of sub-sampled clientsm would thus lead to a choice of ??.

A lower bound on m is, however, hard to estimate.

That is, because data in federated settings is non-IID and contributions from clients might be very distinct.

We therefore define the between clients variance V c as a measure of similarity between clients' updates.

Definition.

Let w i,j define the (i, j)-th parameter in an update of the form w ??? R q??p , at some communication round t. For the sake of clarity, we will drop specific indexing of communication rounds for now.

The variance of parameter (i, j) throughout all K clients is defined as, DISPLAYFORM3 where DISPLAYFORM4 We then define V c as the sum over all parameter variances in the update matrix as, DISPLAYFORM5 Further, the Update scale U s is defined as, DISPLAYFORM6 Algorithm 1 Client-side differentially private federated optimization.

K is the number of participating clients; B is the local mini-batch size, E the number of local epochs, ?? is the learning rate, {??} T t=0is the set of variances for the GM.

{m t } T t=0 determines the number of participating clients at each communication round.

defines the dp we aim for.

Q is the threshold for ??, the probability that -dp is broken.

T is the number of communication rounds after which ?? surpasses Q. B is a set holding client's data sliced into batches of size B 1: procedure SERVER EXECUTION 2:Initialize: w 0 , Accountant( , K) initialize weights and the priv.

accountant 3:for each round t = 1, 2, ... do

?? ??? Accountant(m t , ?? t ) Accountant returns priv.

loss for current round for each client k ??? Z t in parallel do 8: DISPLAYFORM0 DISPLAYFORM1 w ??? w t

for each local Epoch i = 1, 2, ...

E do

In order to test our proposed algorithm we simulate a federated setting.

For the sake of comparability, we choose a similar experimental setup as BID9 did.

We divide the sorted MNIST set into shards.

Consequently, each client gets two shards.

This way most clients will have samples from two digits only.

A single client could thus never train a model on their data such that it reaches high classification accuracy for all ten digits.

We are investigating differential privacy in the federated setting for scenarios of K ??? {100, 1000, 10000}. In each setting the clients get exactly 600 data points.

For K ??? {1000, 10000}, data points are repeated.

For all three scenarios K ??? {100, 1000, 10000} we performed a cross-validation grid search on the following parameters:??? Number of batches per client B??? Epochs to run on each client E??? Number of clients participating in each round m??? The GM parameter ?? In accordance to BID0 we fixed to the value of 8.

During training we keep track of privacy loss using the privacy accountant.

Training is stopped once ?? reaches e ??? 3, e ??? 5, e ??? 6 for 100, 1000 and 10000 clients, respectively.

In addition, we also analyze the between clients variance over the course of training.

In the cross validation grid search we look for those models that reach the highest accuracy while staying below the respective bound on ??.

In addition, when multiple models reach the same accuracy, the one with fewer needed communication rounds is preferred.

TAB3 holds the best models found for K ??? {100, 1000, 10000}. We list the accuracy (ACC), the number of communication rounds (CR) needed and the arising communication costs (CC).

Communication costs are defined as the number of times a model gets send by a client over the course of training, i.e.

T t=0 m t .

In addition, as a benchmark, TAB3 also holds the ACC, CR and CC of the best performing non-differentially private model for K = 100.

In Fig. 2 , the accuracy of all four best performing models is depicted over the course of training.

In FIG3 , the accuracy of non-differentially private federated optimization for K = 100 is depicted again together with the between clients variance and the update scale over the course of training.

100 clients, non differentially private 10000 clients, differentially private 1000 clients, differentially private 100 clients, differentially private Figure 2 : Accuracy of digit classification from non-IID MNIST-data held by clients over the course of decentralized training.

For differentially private federated optimization, dots at the end of accuracy curves indicate that the ??-threshold was reached and training therefore stopped.

As intuitively expected, the number of participating clients has a major impact on the achieved model performance.

For 100 and 1000 clients, model accuracy does not converge and stays significantly below the non-differentially private performance.

However, 78% and 92% accuracy for K ??? {100, 1000} are still substantially better than anything clients would be able to achieve when only training on their own data.

In domains where K lays in this order of magnitude and differential privacy is of utmost importance, such models would still substantially benefit any client participating.

An example for such a domain are hospitals.

Several hundred could jointly learn a model, while information about a specific hospital stays hidden.

In addition, the jointly learned model could be used as an initialization for further client-side training.

For K = 10000, the differentially private model almost reaches accuracies of the non-differential private one.

This suggests that for scenarios where many parties are involved, differential privacy comes at almost no cost in model performance.

These scenarios include mobile phones and other consumer devices.

In the cross-validation grid search we also found that raising m t over the course of training improves model performance.

When looking at a single early communication round, lowering both m t and ?? t in a fashion such that ?? 2 t /m t stays constant, has almost no impact on the accuracy gain during that round.

however, privacy loss is reduced when both parameters are lowered.

This means more communication rounds can be performed later on in training, before the privacy budget is drained.

In subsequent communication rounds, a large m t is unavoidable to gain accuracy, and a higher privacy cost has to be embraced in order to improve the model.

This observation can be linked to recent advances of information theory in learning algorithms.

As observable in FIG3 , BID17 suggest, we can distinguish two different phases of training: label fitting and data fitting phase.

During label fitting phase, updates by clients are similar and thus V c is low, as FIG3 shows.

U c , however, is high during this initial phase, as big updates to the randomly initialized weights are performed.

During data fitting phase V c rises.

The individual updates w k look less alike, as each client optimizes on their data set.

U c however drastically shrinks, as a local optima of the global model is approached, accuracy converges and the contributions cancel each other out to a certain extend.

FIG3 shows these dependencies of V c and U c .We can conclude: i) At early communication rounds, small subsets of clients might still contribute an average update w t representative of the true data distribution ii) At later stages a balanced (and therefore bigger) fraction of clients is needed to reach a certain representativity for an update.

iii) High U c makes early updates less vulnerable to noise.

We were able to show through first empirical studies that differential privacy on a client level is feasible and high model accuracies can be reached when sufficiently many parties are involved.

Furthermore, we showed that careful investigation of the data and update distribution can lead to optimized privacy budgeting.

For future work, we plan to derive optimal bounds in terms of signal to noise ratio in dependence of communication round, data representativity and between-client variance as well as further investigate the connection to information theory.

Additionally, we plan to further investigate the dataset dependency of the bounds.

For assessing further applicability in bandwith-limited settings, we plan to investigate the applicability of proposed approach in context of compressed gradients such as proposed by BID8 .

<|TLDR|>

@highlight

Ensuring that models learned in federated fashion do not reveal a client's participation.