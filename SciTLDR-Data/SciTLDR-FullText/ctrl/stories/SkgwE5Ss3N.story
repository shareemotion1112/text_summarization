Federated learning involves training and effectively combining machine learning models from distributed partitions of data (i.e., tasks) on edge devices, and be naturally viewed as a multi- task learning problem.

While Federated Averaging (FedAvg) is the leading optimization method for training non-convex models in this setting, its behavior is not well understood in realistic federated settings when the devices/tasks are statistically heterogeneous, i.e., where each device collects data in a non-identical fashion.

In this work, we introduce a framework, called FedProx, to tackle statistical heterogeneity.

FedProx encompasses FedAvg as a special case.

We provide convergence guarantees for FedProx through a device dissimilarity assumption.

Our empirical evaluation validates our theoretical analysis and demonstrates the improved robustness and stability of FedProx for learning in heterogeneous networks.

Large networks of remote devices, such as phones, vehicles, and wearable sensors, generate a wealth of data each day.

Federated learning has emerged as an attractive paradigm to push the training of models in such networks to the edge (McMahan et al., 2017) .

In such settings, the goal is to jointly learn over distributed partitions of data/tasks, where statistical heterogeneity and systems constraints present significant challenges.

Optimization methods that allow for local updating and low participation have become the de facto solvers for federated learning (McMahan et al., 2017; Smith et al., 2017) .

These methods perform a variable number of local updates on a subset of devices to enable flexible and efficient communication.

Of current federated optimization methods, FedAvg (McMahan et al., 2017) has become state-of-the-art for non-convex federated learning.

However, FedAvg was not designed to tackle the statistical heterogeneity which is inherent in federated settings; namely, that data may be non-identically distributed across devices.

In realistic statistically heterogeneous settings, FedAvg has been shown to diverge empirically (McMahan et al., 2017, Sec 3) , and it also lacks theoretical convergence guarantees.

Indeed, recent works exploring convergence guarantees are limited to unrealistic scenarios, where (1) the data is either shared across devices or distributed in an IID (identically and independently distributed) manner, or (2) all devices are active at each communication round (Zhou & Cong, 2017; Stich, 2018; Wang & Joshi, 2018; Woodworth et al., 2018; Yu et al., 2018; Wang et al., 2018) .Due to the statistical heterogeneity of the data in federated networks, one can think of federated learning as a prime example of distributed multi-task learning, where each device corresponds to a task.

However, the more common goal of federated learning-and the focus of this work-involves training a single global model on distributed data collected for these various tasks.

We introduce and study a novel optimization framework in the federated setting.

Our focus on its convergence behavior in the face of statistically heterogeneous data is closely related to the classical multi-task setting which involves jointly learning task-specific models from statistically heterogeneous data.

Contributions.

We propose a federated optimization framework for heterogeneous networks, FedProx, which encompasses FedAvg.

In order to characterize the convergence behavior of FedProx, we invoke a device dissimilarity assumption in the network.

Under this assumption, we provide the first convergence guarantees for FedProx.

Finally, we demonstrate that our theoretical assumptions reflect empirical performance, and that FedProx can improve the robustness and stability of convergence over FedAvg when data is heterogeneous across devices.

Large-scale distributed machine learning has motivated the development of numerous distributed optimization meth-ods in the past decade (see, e.g., BID5 Zhang et al., 2013; Li et al., 2014a; Shamir et al., 2014; Reddi et al., 2016; Zhang et al., 2015; Richt??rik & Tak????, 2016; BID3 .

However, it is increasingly attractive to learn statistical models directly over networks of distributed devices.

This problem, known as federated learning, requires tackling novel challenges with privacy, heterogeneous data, and massively distributed networks.

Recent optimization methods have been proposed that are tailored to the specific challenges in the federated setting.

These methods have shown significant improvements over traditional distributed approaches like ADMM BID2 by allowing both for inexact local updating in order to balance communication vs. computation in large networks, and for a small subset of devices to be active at any communication round (McMahan et al., 2017; Smith et al., 2017; Lin et al., 2018) .

For example, Smith et al. (2017) proposes a communication-efficient primal-dual optimization method that learns separate but related models for each device through a multi-task learning framework.

However, such an approach does not generalize to non-convex problems, e.g. deep learning, due to lack of strong duality.

In the non-convex setting, Federated Averaging (FedAvg), a heuristic method based on averaging local Stochastic Gradient Descent (SGD) updates, has instead been shown to work well empirically (McMahan et al., 2017) .Unfortunately, FedAvg is quite challenging to analyze due to its local updating scheme, the fact that few devices are active at each round, and the issue that data is heterogeneous.

Recent works have made steps towards analyzing FedAvg in simpler settings.

For instance, parallel SGD and related variants (Zhang et al., 2015; Zhou & Cong, 2017; Stich, 2018; Wang & Joshi, 2018; Woodworth et al., 2018) , which make local updates similar to FedAvg, have been studied in the IID setting.

Although some works (Yu et al., 2018; Wang et al., 2018; Hao et al., 2019) have recently explored convergence guarantees in heterogeneous settings, they make the limiting assumptions such as full participation of all devices, convexity (Wang et al., 2018) , or uniformly bounded gradients (Yu et al., 2018) .

There are also several heuristic approaches that aim to tackle statistical heterogeneity, either by sharing the local device data or some server-side proxy data (Jeong et al., 2018; Zhao et al., 2018; Huang et al., 2018) , which may be unrealistic in practical federated settings.

In this section, we introduce the key ingredients behind recent methods for federated learning, including FedAvg, and then outline our proposed framework, FedProx.

Federated learning methods (e.g., McMahan et al., 2017; Smith et al., 2017; Lin et al., 2018) are designed to handle multiple devices collecting data and a central server coordinating the global learning objective across the network.

The aim is to minimize: DISPLAYFORM0 where N is the number of devices, p k ??? 0, ???k, and k p k =1.

In general, the local objectives measure the local empirical risk over possibly differing data distributions DISPLAYFORM1 , with n k samples available at each device k. Hence, we can set p k = n k n , where n= k n k is the total number of data points.

To reduce communication and handle systems constraints, federated optimization methods commonly allow for low participation and local updating.

At each round, a subset of the devices are selected and use local solvers to optimize the local objectives.

Then the local updates are aggregated via a central server.

Each of the local objectives can be solved inexactly, as formally defined below.

Definition 1 (??-inexact solution).

For a function h(w; w 0 ) = F (w) + ?? 2 w ??? w 0 2 , and ?? ??? [0, 1], we say w * is a ??-inexact solution of min w h(w; w 0 ), if ???h(w * ; w 0 ) ??? ?? ???h(w 0 ; w 0 ) , where ???h(w; w 0 ) = ???F (w) + ??(w ??? w 0 ).

Note that a smaller ?? corresponds to higher accuracy.

We use ??-inexactness in our analysis (Section 4) to measure the amount of local computation from each local solver.

In experiments (Section 5), we simply run an iterative local solver for some number of local epochs, which can be seen as a proxy for ??-inexactness.

In Federated Averaging (FedAvg) (McMahan et al., 2017) , at each round, a subset K N of devices are selected and run SGD locally for E number of epochs to optimize the local objective F k on device k, and then the resulting model updates are averaged.

McMahan et al. (2017) shows empirically that it is crucial to tune the number of local epochs for FedAvg to converge, as additional local epochs allow local models to move further away from the initial global model, potentially causing divergence.

Thus, it is beneficial to restrict the amount of local deviation through a more principled tool than heuristically limiting the number of local epochs of some iterative solver.

This serves as our inspiration for FedProx, introduced below.

Instead of just minimizing the local function F k , in FedProx, device k uses its local solver to approximately minimize the following surrogate objective h k : DISPLAYFORM0 The proximal term in the above expression effectively limits the impact of local updates by restricting them to be close to the current model w t .

We note that proximal terms such as the one above are a popular tool utilized throughout the optimization literature (see Appendix C).

An important distinction of the proposed usage is that we suggest, explore, and analyze such a term for the purpose of tackling statistical heterogeneity in federated settings.

DISPLAYFORM1 Server selects a subset S t of K devices at random (each device k is chosen with probability p k ); Server sends w t to all chosen devices; Each chosen device k ??? S t finds a w t+1 k which is a ??-inexact minimizer of: DISPLAYFORM2 Each chosen device k sends w t+1 k back to the server; Server aggregates the w's as w DISPLAYFORM3 In Section 4, we see that the usage of the proximal term makes FedProx more amenable to theoretical analysis.

In Section 5, we also see the modified local subproblem in FedProx results in more robust and stable convergence compared to FedAvg for heterogeneous datasets.

Note that FedAvg is a special case of FedProx with ?? = 0.

In this section we first introduce a metric that specifically measures the dissimilarity among local functions.

We call this metric local dissimilarity.

We then analyze FedProx under an assumption on bounded local dissimilarity.

Definition 2 (B-local dissimilarity).

The local functions DISPLAYFORM0 for ???f (w) = 0.Here E k [??] denotes the expectation over devices with masses p k =n k /n and N k=1 p k =1.

Note that B(w)??? 1 and the larger the value of B(w), the larger is the dissimilarity among the local functions.

Moreover, if F k (??)'s are associated with empirical risk objectives and the samples on all the devices are homogeneous, then B(w) ??? 1 for every w as all the local functions converge to the same expected risk function.

Interestingly, similar assumptions (e.g., Vaswani et al., 2019; Yin et al., 2018) have been explored elsewhere for differing purposes; see more in Appendix C. Using Definition 2, we now state our formal dissimilarity assumption, which we use in our convergence analysis.

Assumption 1 (Bounded dissimilarity).

For some > 0, there exists a B such that for all the points w ??? S c = {w | ???f (w) 2 > }, B(w) ??? B .Using Assumption 1, we analyze the amount of expected objective decrease if one step of FedProx is performed.

Theorem 3 (Non-convex FedProx Convergence: B-local dissimilarity).

Let Assumption 1 hold.

Assume the functions F k are non-convex, L-Lipschitz smooth, and there exists DISPLAYFORM1 t is not a stationary solution and the local functions F k are B-dissimilar, i.e. B(w t ) ??? B. If ??, K, and ?? in Algorithm 1 are chosen such that DISPLAYFORM2 then at iteration t of Algorithm 1, we have the following expected decrease in the global objective: DISPLAYFORM3 where S t is the set of K devices chosen at iteration t.

We direct the reader to Appendix A.1 for a detailed proof.

Theorem 3 uses the dissimilarity in Definition 2 to identify sufficient decrease at each iteration for FedProx.

In Appendix A.2, we provide a corollary characterizing the performance with a more common (though slightly more restrictive) bounded variance assumption.

Remark 4.

In order for ?? in Theorem 3 to be positive, we need ??B < 1.

Moreover, we also need DISPLAYFORM4 These conditions help to quantify the trade-off between dissimilarity bound (B) and the algorithm parameters (??, K).Finally, we can use the above sufficient decrease to characterize the rate of convergence under Assumption 1.

Note that these results hold for general non-convex F k (??).Theorem 5 (Convergence rate: FedProx).

Given some > 0, assume that for B ??? B , ??, ?? and K the assumptions of Theorem 3 hold at each iteration of FedProx.

DISPLAYFORM5 While the results thus far hold for non-convex F k (??), we prove the convergence for convex loss in Appendix A.3.

To help provide context for the rate in Theorem 5, we compare it with SGD in the convex case in Appendix A.4, Remark 9.

We now present empirical results for FedProx.

We study the effect of statistical heterogeneity on the convergence of FedAvg and FedProx, explore properties of the FedProx framework, and show how empirical convergence relates to the bounded dissimilarity assumption.

We show a subset of our experiments here due to space constraints; for full results we defer the reader to Appendix B. All code, data, and experiments are publicly available at github.com/litian96/FedProx.

Experimental Details.

We evaluate FedProx on diverse tasks, models, and both synthetic and real-world datasets.

The real datasets are curated from prior work in federated learning (McMahan et al., 2017; BID3 .

In particular, We study convex models on partitioned MNIST (LeCun et al., 1998) , Federated Extended MNIST BID4 BID3 ) (FEM-NIST), and FMNIST*, and non-convex models on Sentiment140 BID7 ) FORMULA0 Effect of Statistical Heterogeneity.

In Figure 1 , we study how statistical heterogeneity affects convergence using four synthetic datasets.

From left to right, as data become more heterogeneous, convergence becomes worse for FedProx with ??=0 (FedAvg).

Setting ?? > 0 is particularly useful in heterogeneous settings although that may slow convergence for IID data.

Properties of FedProx Framework.

The key parameters of FedProx that affect performance are the number of local epochs, E, and the proximal term scaled by ??. We study FedProx under different values of E and ?? using the federated datasets described in TAB3 in Appendix B.1.

We report the results on Shakespeare dataset here and provide similar results on all datasets in Appendix B.3.(1) Dependence on E. We explore the effect of E in Figure 2 (left) and show the convergence in terms of the training loss.

We see that large E leads to divergence on Shakespeare.

In Appendix B.3, we further show that large E leads to similar instability on other heterogeneous datasets.

We note here that a large E may be particularly useful in practice when communication is expensive (which is common in federated networks) where small E is prohibitive.

In Figure 3 , e.g., we show that FedProx with a large E (E=50) and an appropriate ?? (??=0.2) leads to faster and more stable convergence compared with E=1, ??=0 (slow convergence) and E=50, ??=0 (unstable convergence).

Figure 1 .

Effect of data heterogeneity on convergence.

We show training loss (see testing accuracy and dissimilarity metric in Appendix B.3, FIG4 ) on four synthetic datasets whose heterogeneity increases from left to right.

The method with ?? = 0 corresponds to FedAvg.

Increasing heterogeneity leads to worse convergence, but setting ?? > 0 can help to combat this.

(2) Dependence on ??. We consider the effect of ?? on convergence in Figure 2 (middle).

We observe that the appropriate ?? can force divergent methods to converge or increase the stability for unstable methods ( Figure 5 , Appendix B.3), thus making the performance of FedProx less dependent on E. In practice, ?? can be adaptively chosen based on the current performance of the models.

For example, one simple heuristic is to increase ?? when seeing the loss increasing and decreasing ?? when seeing the loss decreasing.

We provide additional experiments demonstrating the effectiveness of this approach in Appendix B.5.Dissimilarity Measurement and Divergence.

Finally, in Figure 2 (right), we track the variance of gradients on each device, DISPLAYFORM0 , which is lower bounded by B (see Bounded Variance Equivalence Corollary 6).

We observe that the dissimilarity metric in Definition 2 is consistent with the training loss.

Therefore, smaller dissimilarity indicates better convergence, which can be enforced by setting ?? appropriately.

Proof.

Using our notion of ??-inexactness for each local solver (Definition 1), we can define e t+1 k such that: DISPLAYFORM1 Now let us definew DISPLAYFORM2 .

Based on this definition, we know DISPLAYFORM3 Let us define?? = ?? ??? L ??? > 0 and?? t+1 k = arg min w h k (w; w t ).

Then, due to the??-strong convexity of h k , we have DISPLAYFORM4 Note that once again, due to the??-strong convexity of h k , we know that DISPLAYFORM5 .

Now we can use the triangle inequality to get DISPLAYFORM6 Therefore, DISPLAYFORM7 where the last inequality is due to the bounded dissimilarity assumption.

Now let us define M t+1 such thatw DISPLAYFORM8 where the last inequality is also due to bounded dissimilarity assumption.

Based on the L-Lipschitz smoothness of f and Taylor expansion, we have DISPLAYFORM9 From the above inequality it follows that if we set the penalty parameter ?? large enough, we can get a decrease in the objective value of f (w t+1 ) ??? f (w t ) which is proportional to ???f (w t ) 2 .

However, this is not the way that the algorithm works.

In the algorithm, we only use K devices that are chosen randomly to approximatew t .

So, in order to find the E f (w t+1 ) , we use local Lipschitz continuity of the function f .

DISPLAYFORM10 where L 0 is the local Lipschitz continuity constant of function f and we have DISPLAYFORM11 Therefore, if we take expectation with respect to the choice of devices in round t we need to bound DISPLAYFORM12 where Q t = E St L 0 w t+1 ???w t+1 .

Note that the expectation is taken over the random choice of devices to update.

DISPLAYFORM13 From FORMULA19 , we have that DISPLAYFORM14 and DISPLAYFORM15 where the first inequality is a result of K devices being chosen randomly to get w t and the last inequality is due to bounded dissimilarity assumption.

If we replace these bounds in (13) we get DISPLAYFORM16 Combining (9), (12), (10) and (16) and using the notation ?? = 1 ?? we get DISPLAYFORM17

Theorem 3 uses the dissimilarity in Definition 2 to identify sufficient decrease at each iteration for FedProx.

Here we provide a corollary characterizing the performance with a more common (though slightly more restrictive) bounded variance assumption.

This assumption is commonly employed, e.g., when analyzing methods such as SGD.

Corollary 6 (Bounded Variance Equivalence).

Let Assumption 1 hold.

Then, in the case of bounded variance, i.e., DISPLAYFORM0 Proof.

We have, DISPLAYFORM1 With Corollary 6 in place, we can restate the main result in Theorem 3 in terms of the bounded variance assumption.

Theorem 7 (Non-Convex FedProx Convergence: Bounded Variance).

Let the assertions of Theorem 3 hold.

In addition, let the iterate w t be such that ???f (w t ) 2 ??? , and let E k ???F k (w) ??? ???f (w) 2 ??? ?? 2 hold instead of the dissimilarity condition.

If ??, K and ?? in Algorithm 1 are chosen such that DISPLAYFORM2 then at iteration t of Algorithm 1, we have the following expected decrease in the global objective: DISPLAYFORM3 where S t is the set of K devices chosen at iteration t.

The proof of Theorem 7 follows from the proof of Theorem 3 by noting the relationship between the bounded variance assumption and the dissimilarity assumption as portrayed by Corollary 6.

Corollary 8 (Convergence: Convex Case).

Let the assertions of Theorem 3 hold.

In addition, let F k (??) be convex and ?? = 0, i.e., all the local problems are solved exactly.

If 1 B ??? 0.5 ??? K, then we can choose ?? ??? 6LB 2 from which it follows that ?? ??? 1 24LB 2 .Proof.

In the convex case, where L ??? = 0 and?? = ??, if ?? = 0, i.e., all subproblems are solved accurately, we can get a decrease proportional to ???f (w t ) 2 if B < ??? K. In such a case if we assume 1 << B ??? 0.5 ??? K, then we can write DISPLAYFORM0 In this case, if we choose ?? ??? 6LB 2 we get DISPLAYFORM1 Note that the expectation in FORMULA0 is a conditional expectation conditioned on the previous iterate.

Taking expectation of both sides, and telescoping, we have that the number of iterations to at least generate one solution with squared norm of gradient less than is O( DISPLAYFORM2

Remark 9 (Comparison with SGD).

Note that FedProx achieves the same asymptotic convergence guarantee as SGD.

In other words, under the bounded variance assumption, for small , if we replace B with its upper-bound in Corollary 6 and choose ?? large enough, then the iteration complexity of FedProx when the subproblems are solved exactly and DISPLAYFORM0 ), which is the same as SGD BID6 .

Synthetic data.

To generate synthetic data, we follow a similar setup to that described in (Shamir et al., 2014) , additionally imposing heterogeneity among devices.

Full details are given in Appendix B.1.

In particular, for each device k, we generate synthetic samples DISPLAYFORM0 , where the covariance matrix ?? is diagonal with ?? j,j = j ???1.2 .

Each element in the mean vector v k is drawn from N (B k , 1), B k ??? N (0, ??).

Therefore, ?? controls how much local models differ from each other and ?? controls how much the local data at each device differs from that of other devices.

We vary ??, ?? to generate three heterogeneous distributed datasets, Synthetic (??, ??), as shown in Figure 1 .

We also generate one IID dataset by setting the same W, b on all devices and setting X k to follow the same distribution.

Our goal is to learn a global W and b.

Real data.

We also explore five real datasets, their statistics summarized in TAB3 in Appendix B.1.

These datasets are curated from prior work in federated learning as well as recent federated learning-related benchmarks (McMahan et al., 2017; BID3 .

We study two convex models on partitioned MNIST (LeCun et al., 1998) , Federated Extended MNIST BID4 BID3 ) (FEMNIST), and FMNIST*. We study two non-convex models on Sentiment140 BID7 ) FORMULA0 Implementation.

We implement FedAvg and FedProx in Tensorflow BID0 .

See details in Appendix B.2.Setup.

For each experiment, we tune the learning rate and ratio of active devices per round on FedAvg.

We randomly split the data on each local device into 80% training set and 20% testing set.

For each comparison, the devices selected and data read at each round are the same across all runs.

We report all metrics based on the global objective f (w).

Note that FedAvg (?? = 0) and FedProx (?? ??? 0) perform the same amount of work at each round when the number of local epochs, E, is the same; we therefore report results in terms of rounds rather than FLOPs or wall-clock time.

Here we provide full details on the datasets and models used in our experiments.

We curate a diverse set of non-synthetic datasets, including those used in prior work on federated learning (McMahan et al., 2017) , and some proposed in LEAF, a benchmark for federated settings BID3 .

We also create synthetic data to directly test the effect of heterogeneity on convergence, as in Section 5.??? Synthetic: We set (??, ??)=(0,0), (0.5,0.5) and (1,1) respectively to generate three non-identical distributed datasets (Figure 1 ).

In the IID data, we set the same W, b ??? N (0, 1) on all devices and X k to follow the same distribution N (v, ??) where each element in the mean vector v is drawn from N (0, 1) and ?? is diagonal with ?? j,j = j ???1.2 .

For all synthetic datasets, there are 30 devices in total and the number of samples on each device follows a power law.??? MNIST: We study image classification of handwritten digits 0-9 in MNIST (LeCun et al., 1998) using multinomial logistic regression.

To simulate a heterogeneous setting, we distribute the data among 1000 devices such that each device has samples of only 2 digits and the number of samples per device follows a power law.

The input of the model is a flattened 784-dimensional (28 ?? 28) image, and the output is a class label between 0 and 9.??? FEMNIST: We study an image classification problem on the 62-class EMNIST dataset BID4 using multinomial logistic regression.

Each device corresponds to a writer of the digits/characters in EMNIST.

We call this federated version of EMNIST FEMNIST.

The input of the model is a flattened 784-dimensional (28 ?? 28) image, and the output is a class label between 0 and 61.??? Shakespeare: This is a dataset built from The Complete Works of William Shakespeare (McMahan et al., 2017) .

Each speaking role in a play represents a different device.

We use a two layer LSTM classifier containing 100 hidden units with a 8D embedding layer.

The task is next character prediction and there are 80 classes of characters in total.

The model takes as input a sequence of 80 characters, embeds each of the character into a learned 8 dimensional space and outputs one character per training sample after 2 LSTM layers and a densely-connected layer.??? Sent140: In non-convex settings, we consider a text sentiment analysis task on tweets from Sentiment140 BID7 ) (Sent140) with a two layer LSTM binary classifier containing 256 hidden units with pretrained 300D GloVe embedding (Pennington et al., 2014) .

Each twitter account corresponds to a device.

The model takes as input a sequence of 25 characters, embeds each of the character into a 300 dimensional space by looking up Glove and outputs one character per training sample after 2 LSTM layers and a densely-connected layer.

??? FEMNIST*: We generate FEMNIST* by subsampling 26 lower case characters from FEMNIST and distributing only 20 classes to each device.

There are 200 devices in total.

The model is the same as the one used on FEMNIST.

We report the total number of devices, samples, and the mean and standard deviation of samples per device of real federated datasets in TAB3 .

(Implementation) In order to draw a fair comparison with FedAvg, we use SGD as a local solver for FedProx, and adopt a slightly different device sampling scheme than that in Algorithms FedAvg and 1: sampling devices uniformly and averaging updates with weights proportional to the number of local data points (as originally proposed in (McMahan et al., 2017) ).

While this sampling scheme is not supported by our analysis, we observe similar relative behavior of FedProx vs. FedAvg whether or not it is employed.

Interestingly, we also observe that the sampling scheme proposed herein results in more stable performance for both methods (see Appendix B.4, Figure 10 ).

This suggests an added benefit of the proposed framework.(Machines) We simulate the federated learning setup (1 server and N devices) on a commodity machine with 2 Intel R Xeon R E5-2650 v4 CPUs and 8 NVidia R 1080Ti GPUs.(Hyperparameters) For each dataset, we tune the ratio of active clients per round from {0.01, 0.05, 0.1} on FedAvg.

For synthetic datasets, roughly 10% of the devices are active at each round.

For MNIST, FEMNIST, Shakespeare, Sent140 and FEMNIST*, the number of active devices (K) are 1%, 5%, 10%, 1% and 5% respectively.

We also do a grid search on the learning rate based on FedAvg.

We do not decay the learning rate through all rounds.

For all synthetic data experiments, the learning rate is 0.01.

For MNIST, FEMNIST, Shakespeare, Sent140 and FEMNIST*, we use the learning rates of 0.03, 0.003, 0.8, 0.3 and 0.003.

We use a batch size of 10 for all experiments.(Libraries) All code is implemented in Tensorflow BID0 Version 1.10.1.

Please see github.com/litian96/FedProx for full details.

We explore the effect of E in Figure 4 .

For each dataset, we set E to be 1, 20, and 50 while keeping ?? = 0 (FedProx reduces to FedAvg in this case) and show the convergence in terms of the training loss.

We see that large E leads to divergence or instability on MNIST and Shakespeare.

On FEMNIST and Sent140, nevertheless, larger E speeds up the convergence.

Based on conclusions drawn from Figure 1 , we hypothesize this is due to the fact that the data distributed across devices after partitioning FEMNIST and Sent140 lack significant heterogeneity.

We validate this hypothesis by observing instability on FEMNIST*, which is a skewed variant of the FEMNIST dataset.

We consider the effect of ?? on convergence in Figure 5 .

For each experiment, in the case of E = 50, we compare the results between ?? = 0 and the best ??. For three out of the four datasets (all but Sent140) we observe that the appropriate ?? can increase the stability for unstable methods and can force divergent methods to converge.

Finally, in Figure 6 , we demonstrate that our B-local dissimilarity measurement in Definition 2 captures the heterogeneity of datasets and is therefore an appropriate proxy of performance.

In particular, we track the variance of gradients on each device, DISPLAYFORM0 , which is lower bounded by B (see Bounded Variance Equivalence Corollary 6).

We observe that the dissimilarity metric is consistent with the training loss.

Therefore, smaller dissimilarity indicates better convergence, which can be enforced by setting ?? appropriately.

Full results tracking B (for all experiments performed) are provided in Appendix B.3.We present testing accuracy, training loss and dissimilarity measurements of all the experiments in FIG4 , Figure 8 and Figure 9 .

We show the training loss, testing accuracy and dissimilarity measurement of FedProx using two different device sampling schemes in Figure 10 .

We show a simple adaptive heuristic of setting ?? on four synthetic datasets in Figure 11 .

Two aspects of the proposed work: our framework, FedProx, and analysis tool, the bounded dissimilarity assumption, have been utilized throughout the optimization literature-though often with very different motivations.

For completeness, we provide a discussion below on our relation to these prior works.

Figure 9 .

Training loss, testing accuracy and dissimilarity measurement for experiments in Figure 5 Proximal term.

We note here a connection to elastic averaging SGD (EASGD) (Zhang et al., 2015) , which was proposed as a way to train deep networks in the data center setting, and uses a similar proximal term in its objective.

While the intuition is similar to EASGD (this term helps to prevent large deviations on each device/machine), EASGD employs a more complex moving average to update parameters, is limited to using SGD as a local solver, and has only been analyzed Figure 10 .

Differences between two sampling schemes in terms of training loss, testing accuracy and dissimilarity measurement.

Sampling devices with a probability proportional to the number of local data points and then simply averaging local models performs slightly better than uniformly sampling devices and averaging the local models with weights proportional to the number of local data points.

Under either sampling scheme, the settings with ?? = 1 demonstrate more stable performance than settings with ?? = 0.for simple quadratic problems.

The proximal term we introduce has also been explored in previous optimization literature with very different purposes, such as (Allen-Zhu, 2018), to speed up (mini-batch) SGD training on a single machine.

Li et al. (2014b) also employs a similar proximal term for efficient SGD training both in a single machine and distributed settings, but their analysis is limited to a single machine setting with different assumptions (e.g., IID data and solving the subproblem exactly at each round).

DANE (Shamir et al., 2014) also includes a proximal term in the local objective function.

However, due to the inexact estimation of full gradients (i.e., ?????(w (t???1) ) in (Shamir et al., 2014, Eq (13) )) with device subsampling schemes and the staleness of the gradient correction term (Shamir et al., 2014, Eq (13) ) in local updating methods, it is not directly applicable to our setting and performs worse on heterogeneous datasets (see Figure 12) .Bounded dissimilarity assumption.

The bounded dissimilarity assumption has appeared in different forms, for example in (Yin et al., 2018; Vaswani et al., 2019) .

In (Yin et al., 2018) , the bounded similarity assumption is used in context of asserting gradient diversity and quantifying the benefit in terms of scaling of the mean square error for mini-batch SGD for data which is i.i.d.

In (Vaswani et al., 2019) , the authors use a similar assumption, called strong growth condition, which is a stronger version of Assumption 1 with = 0.

They prove that some interesting practical problems satisfy such a condition.

They also use this assumption to prove better convergence rates for SGD with constant step-size.

Note that this is different with our approach as the algorithm that we are analyzing is not SGD and our analysis is different in spite of the similarity in the assumptions.

<|TLDR|>

@highlight

We introduce FedProx, a framework to tackle statistical heterogeneity in federated settings with convergence guarantees and improved robustness and stability.