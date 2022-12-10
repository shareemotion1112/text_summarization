Federated learning involves jointly learning over massively distributed partitions of data generated on remote devices.

Naively minimizing an aggregate loss function in such a network may disproportionately advantage or disadvantage some of the devices.

In this work, we propose q-Fair Federated Learning (q-FFL), a novel optimization objective inspired by resource allocation strategies in wireless networks that encourages a more fair accuracy distribution across devices in federated networks.

To solve q-FFL, we devise a scalable method, q-FedAvg, that can run in federated networks.

We validate both the improved fairness and flexibility of q-FFL and the efficiency of q-FedAvg through simulations on federated datasets.

With the growing prevalence of IoT-type devices, data is frequently collected and processed outside of the data center and directly on distributed devices, such as wearable devices or mobile phones.

Federated learning is a promising learning paradigm in this setting that pushes statistical model training to the edge (McMahan et al., 2017) .The number of devices in federated networks is generally large-ranging from hundreds to millions.

While one can naturally view federated learning as a multi-task learning problem where each device corresponds to a task (Smith et al., 2017) , the focus is often instead to fit a single global model over these distributed devices/tasks via some empirical risk minimization objective (McMahan et al., 2017) .

Naively minimizing the average loss via such an objective may disproportionately advantage or disadvantage some of the devices, which is exacerbated by the fact that the data are often heterogeneous across devices both in terms of size and distribution.

In this work, we therefore ask: Can we devise an efficient optimization method to encourage a more fair distribution of the model performance across devices in federated networks?There has been tremendous recent interest in developing fair methods for machine learning.

However, current methods that could help to improve the fairness of the accuracy distribution in federated networks are typically proposed for a much smaller number of devices, and may be impractical in federated settings due to the number of involved constraints BID5 .

Recent work that has been proposed specifically for the federated setting has also only been applied at small scales (2-3 groups/devices), and lacks flexibility by optimizing only the performance of the single worst device (Mohri et al., 2019) .In this work, we propose q-FFL, a novel optimization objective that addresses fairness issues in federated learning.

Inspired by work in fair resource allocation for wireless networks, q-FFL minimizes an aggregate reweighted loss parameterized by q such that the devices with higher loss are given higher relative weight to encourage less variance in the accuracy distribution.

In addition, we propose a lightweight and scalable distributed method, qFedAvg, to efficiently solve q-FFL, which carefully accounts for important characteristics of the federated setting such as communication-efficiency and low participation of devices BID3 McMahan et al., 2017) .

We empirically demonstrate the fairness, efficiency, and flexibility of q-FFL and q-FedAvg compared with existing baselines.

On average, q-FFL is able to reduce the variance of accuracies across devices by 45% while maintaining the same overall average accuracy.

Fairness in Machine Learning.

There are several widespread approaches in the machine learning community to address fairness, which is typically defined as the protection of some specific attribute(s) (e.g., (Hardt et al., 2016) ).

In addition to preprocess the data and post-process the model BID8 Hardt et al., 2016) , another set of works optimize an objective under some explicit fairness constraints during training time BID2 BID5 Hashimoto et al., 2018; Woodworth et al., 2017; Zafar et al., 2017; BID8 .

Our work also enforces fairness during training, though we define fairness as the accuracy distribution across devices in federated learning, as opposed to the protection of a specific attribute (Section 3).

BID5 use a notion of 'minimum accuracy' as one special case of the rate constraints, which is conceptually similar to our goal.

However, it requires each device to have one constraint, which is not practical in the federated setting.

In federated settings, Mohri et al. (2019) proposes a minimax optimization scheme, Agnostic Federated Learning (AFL), which optimizes for the performance of the single worst device.

This method has only been applied at small scales (for a handful of groups).

In addition, our objective is more flexible because q may be tuned based on the amount of fairness desired.

Fairness in Resource Allocation.

Fair resource allocation has been extensively studied in fields such as network management BID6 Hahne, 1991; Kelly et al., 1998; Neely et al., 2008) and wireless communications BID7 Nandagopal et al., 2000; Sanjabi et al., 2014; Shi et al., 2014) .

In these contexts, the problem is defined as allocating a scarce shared resource, e.g. communication time or power, among many users.

In these cases directly maximizing utilities such as total throughput usually leads to unfair allocations where some users receive poor service.

Several measurements have been proposed to balance between fairness and total throughput.

Among them, a unified framework is captured through α-fairness (Lan et al., 2010; Mo & Walrand, 2000) , in which the emphasis on fairness can be tuned by changing a single parameter, α.

If we think of the global model as a resource to serve the users (or devices), it is natural to ask similar questions about the fairness of the service that devices receive and use similar tools to promote fairness.

Despite this, we are unaware of any work that uses fairness criteria from resource allocation to modify training objectives in machine learning.

Inspired by the α-fairness metric, we propose a similarly modified objective function, q-Fair Federated Learning (q-FFL), to encourage a more fair accuracy distribution across devices in the context of federated training.

We empirically demonstrate its benefits in Section 4.Federated and Distributed Optimization.

Federated learning faces fundamental challenges such as expensive communication, variability in hardware, network connection, and power of devices, and heterogeneous local data distribution amongst devices, making it distinct from classical distributed optimization (Recht et al., 2011; ShalevShwartz & Zhang, 2013; BID4 .

In order to reduce communication, as well as to tolerate heterogeneity, methods that allow for local updating and low participation among devices have become de facto solvers for this setting BID4 McMahan et al., 2017; Smith et al., 2017) .

We incorporate recent advancements in this field when designing methods to solve the q-FFL objective, which we describe in Section 3.3.

We first formally define the classical federated learning objective and methods, and introduce our proposed notion of fairness in Section 3.1.

We then introduce q-FFL, a novel objective that encourages a more fair accuracy distribution across all devices (Section 3.2).

Finally, in Section 3.3, we describe q-FedAvg, an efficient distributed method we develop to solve the objective in federated settings.

Federated learning involves fitting a global model on distributed data generated on hundreds to millions of remote devices.

In particular, the goal is to minimize: DISPLAYFORM0 where m is the total number of devices, p k ≥ 0, and DISPLAYFORM1 The local objective F k 's can be defined by empirical risks over local data, i.e., DISPLAYFORM2 where n k is the number of samples available locally.

We can set p k to be n k n , where n = k n k is the total number of samples in the entire dataset.

Most prior work solves (1) by first subsampling devices with probabilities proportional to n k at each round, and then applying an optimizer such as Stochastic Gradient Descent (SGD) locally to perform updates.

These local updating methods enable flexible and efficient communication by running the optimizer for a variable number of iterations locally on each device, e.g., compared to traditional distributed (stochastic) gradient descent, which would simply calculate a subset of the gradients (Stich, 2019; Wang & Joshi, 2018; Woodworth et al., 2018; Yu et al., 2019) .

FedAvg (Algorithm 2, Appendix A) (McMahan et al., 2017) is one of the leading methods to solve (1).However, solving (1) in this manner can implicitly introduce unfairness among different devices.

For instance, the learned model may be biased towards the devices with higher number of data points.

Formally, we define our desired fairness criteria for federated learning below.

Definition 1 (Fairness of distribution).

For trained models w andw, we say that model w provides a more fair solution to Objective (1) than modelw if the variance of the performance of model w on the m devices, {a 1 , . . .

a m }, is smaller than the variance of the performance of modelw on the m devices, i.e., Var(a 1 , . . .

, a m ) ≤ Var(ã 1 , . . . ,ã m ).In this work, we take 'performance' for device k, a k , to be the testing accuracy of applying the trained model w on the test data for device k. Our goal is to reduce the variance while maintaining the same (or similar) average accuracy.

A natural idea to achieve fairness as defined in (1) would be to reweight the objective-assigning higher weight to devices with poor performance, so that the distribution of accuracies in the network reduces in variance.

Note that this re-weighting must be done dynamically, as the performance of the devices depends on the model being trained, which cannot be evaluated a priori.

Drawing inspiration from α-fairness, a utility function used in fair resource allocation in wireless networks, we propose the following objective q-FFL.

For given local non-negative cost functions F k and parameter q > 0, we define the overall q-Fair Federated Learning (q-FFL) objective as DISPLAYFORM0 Intuitively, the larger we set q, the larger relative price we pay for devices k with high local empirical loss, DISPLAYFORM1 Here, q is a tunable parameter that depends on the amount of fairness we wish to impose in the network.

Setting q = 0 does not encourage any fairness beyond the classical federated learning objective (1).

A larger q means that we emphasize devices with higher losses (lower accuracies), thus reducing the variance between the accuracy distribution and potentially inducing more fairness in accordance with Definition 1.3.3.

The solver: FedAvg-style q-Fair Federated Learning (q-FedAvg)We first propose a fair but less efficient method q-FedSGD, to illustrate the main techniques we use to solve the q-FFL objective (2).

We then provide a more efficient counterpart q-FedAvg, by considering key properties of federated algorithms such as local updating schemes.

Hyperparameter tuning: q and step-sizes.

In devising a method to solve q-FFL (2), we begin by noting that it is important to first determine how to set q. In practice, q can be tuned based on the desired amount of fairness.

It is therefore common to train a family of objectives for different q values so that a practitioner can explore the tradeoff between accuracy and fairness for the application at hand.

Nevertheless, to optimize q-FFL in a scalable fashion, we rely on gradient-based methods, where the stepsize inversely depends on the Lipchitz constant of the function's gradient, which is often unknown and selected via grid search BID10 Nesterov, 2013) .

As we intend to optimize q-FFL for various values of q, the Lipchitz constant will change as we change q-requiring step-size tuning for all values of q. This can quickly cause the search space to explode.

To overcome this issue, we propose estimating the local Lipchitz constant of the gradient for the family of q-FFL by using the Lipchitz constant we infer on q = 0.

This allows us to dynamically adjust the step-size for the q-FFL objective, avoiding the manual tuning for each q. In Lemma 2 we formalize the relation between the Lipschitz constant, L, for q = 0 and q > 0.Lemma 2.

If the non-negative function f (·) has a Lipchitz gradient with constant L, then for any q ≥ 0 and at any point w, DISPLAYFORM2 is an upper-bound for the local Lipchitz constant of the gradient of DISPLAYFORM3 Furthermore, the gradient DISPLAYFORM4 See proof in Appendix B.A first approach: q-FedSGD.

In our first fair algorithm qFedSGD, we solve Objective (2) using mini-batch SGD on a subset of devices at each round, and apply the above result to each selected device to obtain local Lipchitz constants for gradients of local functions F k .

By averaging those estimates, we obtain an estimate for the Lipchitz constant for the gradient of q-FFL.

Then, the step-size (inverse of this estimate) is applied, like other gradient based algorithms; see Algorithm 3 in Appendix A for more details.

Algorithm 1 q-FedAvg (proposed method) DISPLAYFORM5 Server chooses a subset S t of K devices at random (each device k is chosen with prob.

p k )

Server sends w t to all chosen devices 5:Each device k updating w t for E epochs of SGD with step size η to obtainw to run some number of local updates and then apply the updates in the gradient computation of q-FFL.

The details of our method (q-FedAvg) are given in Algorithm 1.

Note that when q=0, q-FFL corresponds to the normal objective in federated learning (Equation FORMULA0 ), and q-FedAvg is also reduced to FedAvg (McMahan et al., 2017) where no fairness modification is introduced.

We first describe our experimental setup, then demonstrate the improved fairness of the q-FFL objective by comparing q-FFL with several baselines, and finally show the efficiency of q-FedAvg compared with q-FedSGD.Experiment setups.

We explore both convex and nonconvex models on four federated datasets curated from prior work in federated learning (Smith et al., 2017; BID4 BID4 .

Full details of the datasets and models are given in Appendix D. Throughout the experiments, we show results on the Vehicle dataset.

Similar results on all datasets are provided in Appendix C. Fairness of q-FFL.

We verify that the proposed objective q-FFL leads to more fair solutions (according to Definition 1) for federated data, compared with FedAvg and two other baselines that are likely to impose fairness in federated networks.(1) Compare with FedAvg.

In Figure 1 (left), we compare the final testing accuracy distributions of two objectives (q=0 and a tuned value of q=5) averaged across 5 random shuffles of Vehicle.

We observe that the objective with q=5 results in more centered (i.e., fair) testing accuracy distributions with lower variance.

We further report the worst and best 10% testing accuracies and the variances of accuracies in Table 1 .

We see that the average testing accuracy remains almost unchanged with the proposed objective despite significant reductions in variance.

See similar results on other datasets in Figure 2 and TAB2 in Appendix C.(2) Compare with weighing each device equally.

We compare q-FFL with a heuristic that samples devices uniformly and report testing accuracy in Figure 1 (middle).

A table with the statistics of accuracy distribution on all datasets is given in the appendix in Table 3 .

While the 'weighing each device equally' heuristic tends to outperform our method in training accuracy distributions ( Figure 5 and TAB8 in Appendix D.3), our method produces more fair solutions in terms of testing accuracies.

One explanation for this is that uniform sampling is a static method and can easily overfit to devices with very few data points, whereas q-FFL has better generalization properties due to its dynamic nature.

FORMULA0 ).

Middle: Fairness of q-FFL q>0 compared with the uniform sampling baseline.

Right: q-FedAvg converges faster than q-FedSGD.

Table 1 .

Statistics of the testing accuracy distribution for q-FFL on Vehicle.

By setting q > 0, the variance of the final accuracy distribution decreases, and the worst 10% accuracy increases, while the overall accuracy remains fairly constant.

Avg.

Worst 10% Best 10% Var.

q = 0 87.3% 43.0%95.7% 291 q = 5 87.7% 69.9%94.0% 48performance of the device with the highest loss.

This is the only work we are aware of that aims to address fairness issues in federated learning.

See Appendix D.2 for details of our AFL implementation.

We also observe that q-FFL outperforms AFL when q is set appropriately TAB4 , Appendix D).

We note that q is also tunable depending on the amount of fairness desired.

Interestingly, we observe q-FFL converges faster compared with AFL (see Figure 7 in Appendix D.3) in terms of communication rounds.

Choosing q. A natural question is determine how q should be tuned in the q-FFL objective.

The framework is flexible in that it allows one to choose q to tradeoff between reduced variance of the accuracy distribution and a high average accuracy.

In particular, a reasonable approach in practice would be to run Algorithm 1 with multiple q's in parallel to obtain multiple final global models, and then let each device select amongst these based on performance on the validation data.

We show benefits of this device-specific strategy in TAB9 in Appendix D.3.Efficiency of q-FedAvg.

Finally, we show the efficiency of q-FedAvg by comparing Algorithm 1 with its nonlocal-updating baseline q-FedSGD (Algorithm 3) with the same objective (q > 0).

At each communication round, q-FedAvg runs one epoch of local updates on each selected device, while q-FedSGD runs gradient descent using all local training data on that device.

In Figure 1 (

We summarize the FedAvg algorithm proposed in (McMahan et al., 2017) below.

Server chooses a subset S t of K devices at random (each device k is chosen with probability p k ) Server sends w t to all chosen devices Each device k updates w t for E epochs of SGD on F k with step-size η to obtain w t+1 kEach chosen device k sends w t+1 k back to the server Server aggregates the w's as w DISPLAYFORM0 We summarize our proposed method q-FedSGD below.

Algorithm 3 q-FedSGD DISPLAYFORM1 Server chooses a subset S t of K devices at random (each device k is chosen with prob.

p k )

Server sends w t to all chosen devices 5:Each device computes:

DISPLAYFORM0 Each chosen device k sends ∆ Server aggregates the computes w t+1 as: DISPLAYFORM1

Proof.

At any point w, we can compute ∇ 2 f (w) DISPLAYFORM0 As a result, DISPLAYFORM1

Fairness of q-FFL.

We demonstrate the improved fairness of q-FFL on all the four datasets in Figure 2 and TAB2 .Comparison with uniform sampling.

We compare q-FFL with uniform sampling schemes and report testing accuracy on all datasets in Figure 3 .

A table with the final accuracies and variances is given in Table 3 .

While the 'weighing each device equally' heuristic tends to outperform our method in training accuracy distributions ( Figure 5 and TAB8 ), our Figure 2 .

q-FFL leads to fairer test accuracy distributions.

With q > 0, the distributions shift towards the center as low accuracies increase at the cost of decreasing high accuracies on some devices.

Setting q=0 corresponds to the original objective (Equation FORMULA0 ).

The selected q values for q > 0 on the four datasets, as well as distribution statistics, are shown in TAB2 .

Figure 3.

q-FFL (q > 0) compared with uniform sampling.

In terms of testing accuracy, our objective produces more fair solutions than uniform sampling.

Distribution statistics are provided in Table 3 .

Table 3 .

More statistics indicating the resulting fairness of q-FFL compared with the uniform sampling baseline.

Again, we observe that the testing accuracy of the worst 10% devices tends to increase, and the variance of the final testing accuracies is smaller.

Figure 4 .

Fix an objective (i.e., using the same q) for each dataset, q-FedAvg (Algorithm 1) compared with q-FedSGD (Algorithm 3).

We can see that our method adopting local updating schemes converges faster in terms of communication rounds on most datasets.method produces more fair solutions in terms of testing accuracies.

One explanation for this is that uniform sampling is a static method and can easily overfit to devices with very few data points, whereas q-FFL has better generalization properties due to its dynamic nature.

Comparison with weighing each device adversarially.

We show the results of comparing q-FFL with AFL on the two datasets in TAB4 .

q-FFL outperforms AFL in terms of increasing the lowest accuracies.

In addition, q-FFL is more flexible as the parameter q enables the trade-off between increasing the worst accuracies and decreasing the best accuracies.

Efficiency of q-FedAvg.

In Figure 4 , we show that on most datasets, q-FedAvg converges faster than q-FedSGD in terms of communication rounds due to its local updating scheme.

We note here that number of rounds is a reasonable metric for comparison between these methods as they process the same amount of data and perform equivalent amount of communication at each round.

Our method is also lightweight, and can be easily integrated into existing implementations of federated learning algorithms such as TensorFlow Federated (TFF).

We provide full details on the datasets and models used in our experiments.

The statistics of four federated datasets are summarized in TAB6 .

We report total number of devices, total number of samples, and mean and deviation in the sizes of total data points on each device.

Additional details on the datasets and models are described below.• Synthetic: We follow a similar set up as that in (Shamir et al., 2014) and impose additional heterogeneity.

The model is y = argmax(softmax(W x + b)), x ∈ R 60 , W ∈ R 10×60 , b ∈ R 10 , and the goal is to learn a global W and b. Samples (X k , Y k ) and local models on each device k satisfies DISPLAYFORM0 , where the covariance matrix Σ is diagonal with Σ j,j = j −1.2 .

Each element in v k is drawn from TAB2 Fair Resource Allocation in Federated Learning N (B k , 1), B k ∼ N (0, 1).

There are 100 devices in total and the number of samples on each devices follows a power law.• Vehicle 1 : We use the same Vehicle Sensor (Vehicle) dataset as (Smith et al., 2017) , modelling each sensor as a device.

Each sample has a 100-dimension feature and a binary label indicating whether this sample is on an AAV-type or DWtype vehicle.

We train a linear SVM.

We tune the hyperparameters in SVM and report the best configuration.• Sent140: It is a collection of tweets from Sentiment140 BID10 ) (Sent140).

The task is text sentiment analysis which we model as a binary classification problem.

The model takes as input a 25-word sequence, embeds each word into a 300-dimensional space using pretrained Glove (Pennington et al., 2014) , and outputs a binary label after two LSTM layers and one densely-connected layer.• We implement a non-stochastic version of AFL where all devices are selected and updated each round and do a grid search on the AFL hyperparameters, γ w and γ λ .

In order to draw a fair comparison, we modify Algorithm 1 by sampling all devices and letting each of them run gradient descent at each round.

We use the same public datasets (Adult and Fashion MNIST) as in (Mohri et al., 2019) .

We randomly split data on each local device into 80% training set, 10% testing set, and 10% validation set.

We tune q from {0.001, 0.01, 0.1, 1, 2, 5, 10, 15} on the validation set and report accuracy distributions on the testing set.

For each dataset TAB2 Fair Resource Allocation in Federated Learning Table 6 .

Average testing accuracy under q-FFL objectives.

We show that the resulting solutions of q=0 and q¿0 objectives have approximately the same accuracies both with respect to all data points and with respect to all devices.

Objective Accuracy w.r.

Figure 5 .

q-FFL (q > 0) compared with uniform sampling in training accuracy.

We see that in most cases uniform sampling has higher (and more fair) training accuracies due to the fact that it is overfitting to devices with few samples.we repeat this process for five randomly selected train/test/validation splits, and report the mean and standard deviation across these five runs where applicable.

For Synthetic, Vehicle, Sent140, and Shakespeare, optimal 2 q values are 1, 5, 1, and 0.001 respectively.

We randomly sample 10 devices each round.

We tune the learning rate on FedAvg and use the same learning rate for all experiments of that dataset.

The learning rates for Synthetic, Vehicle, Sent140, and Shakespeare are 0.1, 0.01, 0.03, and 0.8 respectively.

When running AFL methods, we search for a best γ w and γ λ such that AFL achieves the highest testing accuracy on the device with the highest loss within a fixed number of rounds.

For Adult, we use γ w = 0.1 and γ λ = 0.1; for Fashion MNIST, we use γ w = 0.001 and γ λ = 0.01.

We use the same γ w as step sizes for q-FedAvg on Adult and Fashion MNIST.

In TAB4 , q 1 = 0.01, q 2 = 2 for q-FFL on Adult and q 1 = 5, q 2 = 15 for q-FFL on Fashion MNIST.

The number of local epochs is fixed to 1 whenever we do local updates.

Average testing accuracy with respect to devices.

We have shown that q-FFL leads to more fair accuracy distributions while maintaining approximately the same testing accuracies in Section 4.

Note that we report average testing accuracy with respect to all data points in Table 1 and 2.

We observe similar results on average accuracy with respect to all devices between q = 0 and q > 0 objectives, as shown in Table 6 .Efficiency of q-FFL compared with AFL.

One added benefit of q-FFL is that it leads to faster convergence than AFL even when we use non-local-updating methods for both objectives.

In Figure 7 , we show that when fixing the final testing accuracy for the single worst device, q-FFL converges faster than AFL.

As the number of devices increases (from Fashion MNIST to Vehicle), the performance gap between AFL and q-FFL becomes larger because AFL introduces larger variance TAB2 Fair Resource Allocation in Federated Learning Figure 6 .

The convergence speed of q-FFL compared with FedAvg.

We plot the distance to the highest accuracy achieved versus communication rounds.

Although q-FFL with q > 0 is a more difficult optimization problem, for the q values we choose that could lead to more fair results, the convergence speed is comparable to that of q = 0.Choosing q.

We solve q-FFL with q ∈ {0, 0.001, 0.01, 0.1, 1, 2, 5, 10} in parallel.

After training, each device selects the best resulting model based on the validation data and tests the performance of the model using testing set.

We report the results in terms of testing accuracy in TAB9 .

Using this strategy, accuracy variance is reduced and average accuracy is increased.

However, this will induce more local computation and additional communication load in each round.

But this does not increase the number of communication rounds.

Convergence speed of q-FFL.

In Section 4, we show that our solver q-FedAvg using local updating schemes converges significantly faster than q-FedSGD.

A natural question one might ask is: will the q-FFL (q > 0) objective slows the convergence compared with FedAvg?

We empirically investigate this on real datasets.

We use q-FedAvg to solve q-FFL, and compare it with FedAvg.

As demonstrated in Figure 6 , the q values we are choosing that result in more fair solutions do not significantly slowdown convergence.

52.1% ± .3% 42.1% ± 2.1% 69.0% ± 4.4% 54 ± 27 multiple q's 52.0 ± 1.5% % 41.0% ± 4.3% 72.0% ± 4.8% 72 ± 32

<|TLDR|>

@highlight

We propose a novel optimization objective that encourages fairness in heterogeneous federated networks, and develop a scalable method to solve it.