Federated learning distributes model training among a multitude of agents, who, guided by privacy concerns, perform training using their local data but share only model parameter updates, for iterative aggregation at the server.

In this work, we explore the threat of model poisoning attacks on federated learning initiated by a single, non-colluding malicious agent where the adversarial objective is to cause the model to misclassify a set of chosen inputs with high confidence.

We explore a number of strategies to carry out this attack, starting with simple boosting of the malicious agent's update to overcome the effects of other agents' updates.

To increase attack stealth, we propose an alternating minimization strategy, which alternately optimizes for the training loss and the adversarial objective.

We follow up by using parameter estimation for the benign agents' updates to improve on attack success.

Finally, we use a suite of interpretability techniques to generate visual explanations of model decisions for both benign and malicious models and show that the explanations are nearly visually indistinguishable.

Our results indicate that even a highly constrained adversary can carry out model poisoning attacks while simultaneously maintaining stealth, thus highlighting the vulnerability of the federated learning setting and the need to develop effective defense strategies.

Federated learning introduced by BID11 has recently emerged as a popular implementation of distributed stochastic optimization for large-scale deep neural network training.

It is formulated as a multi-round strategy in which the training of a neural network model is distributed between multiple agents.

In each round, a random subset of agents, with local data and computational resources, is selected for training.

The selected agents perform model training and share only the parameter updates with a centralized parameter server, that facilitates aggregation of the updates.

Motivated by privacy concerns, the server is designed to have no visibility into an agents' local data and training process.

The aggregation algorithm is agnostic to the data distribution at the agents.

In this work, we exploit this lack of transparency in the agent updates, and explore the possibility of a single malicious agent performing a model poisoning attack.

The malicious agent's objective is to cause the jointly trained global model to misclassify a set of chosen inputs with high confidence, i.e., it seeks to introduce a targeted backdoor in the global model.

In each round, the malicious agent generates its update by optimizing for a malicious objective different than the training loss for federated learning.

It aims to achieve this by generating its update by directly optimizing for the malicious objective.

However, the presence of a multitude of other agents which are simultaneously providing updates makes this challenging.

Further, the malicious agent must ensure that its update is undetectable as aberrant.

Contributions: To this end, we propose a sequence of model poisoning attacks, with the aim of achieving the malicious objective while maintaining attack stealth.

For each strategy, we consider both attack strength as well as stealth.

We start with malicious update boosting, designed to negate the combined effect of the benign agents, which enables the adversary to achieve its malicious objective with 100% confidence.

However, we show that boosted updates can be detected as aberrant using two measures of stealth, accuracy checking on the benign objective and parameter update statistics.

Observing that the only parameter updates that need to be boosted are those that con-tribute to the malicious objective, we design an alternating minimization strategy that improves attack stealth.

This strategy alternates between training loss minimization and the boosting of updates for the malicious objective and is able to achieve high success rate on both the benign and malicious objectives.

In addition, we show that estimating the other agents' updates improves attack success rates.

Finally, we use a suite of interpretability techniques to generate visual explanations of the decisions made by a global model with and without a targeted backdoor.

Interestingly, we observe that the explanations are nearly visually indistinguishable.

This establishes the attack stealth along yet another axis of measurement and indicates that backdoors can be inserted without drastic changes in model focus at the input.

In our experiments, we consider adversaries which only control a single malicious agent and at a given time step, have no visibility into the updates that will be provided by the other agents.

We demonstrate that these adversaries can influence the global model to misclassify particular examples with high confidence.

We work with both the Fashion- MNIST Xiao et al. (2017) and Adult Census 1 , datasets and for settings with both 10 and 100 agents, our attacks are able to ensure the global model misclassifies a particular example in a target class with 100% confidence.

Our alternating minimization attack further ensures that the global model converges to the same test set accuracy as the case with no adversaries present.

We also show that a simple estimation of the benign agents' updates as being identical over two consecutive rounds aids in improving attack success.

Related Work: While data poisoning attacks BID1 BID16 BID12 BID22 BID12 BID9 BID4 BID7 have been widely studied, model poisoning attacks are largely unexplored.

A number of works on defending against Byzantine adversaries consider a threat model where Byzantine agents send arbitrary gradient updates BID2 BID5 BID13 BID23 .

However, the adversarial goal in these cases is to ensure a distributed implementation of the Stochastic Gradient Descent (SGD) algorithm converges to 'suboptimal to utterly ineffective models', quoting from BID13 .

In complete constrast, our goal is to ensure convergence to models that are effective on the test set but misclassify certain examples.

In fact, we show that the Byzantine-resilient aggregation mechanism 'Krum BID2 is not resilient to our attack strategies (Appendix C).

Concurrent work by BID0 considers multiple colluding agents performing poisoning via model replacement at convergence time.

In contrast, our goal is to induce targeted misclassification in the global model by a single malicious agent even when it is far from convergence while maintaining its accuracy for most tasks.

In fact, we show that updates generated by their strategy fail to achieve either malicious or benign objectives in the settings we consider.

In this section, we formulate both the learning paradigm and the threat model that we consider throughout the paper.

Operating in the federated learning paradigm, where model weights are shared instead of data, gives rise to the model poisoning attacks that we investigate.

The federated learning setup consists of K agents, each with access to data D i , where DISPLAYFORM0 The total number of samples is i l i = l. Each agent keeps its share of the data (referred to as a shard) private, i.e. DISPLAYFORM1 li } is not shared with the server S. The objective of the server is to learn a global parameter vector w G ∈ R n , where n is the dimensionality of the parameter space.

This parameter vector minimizes the loss 2 over D = ∪ i D i and the aim is to generalize well over D test , the test data.

Federated learning is designed to handle non-i.i.d partitioning of training data among the different agents.

Traditional poisoning attacks deal with a malicious agent who poisons some fraction of the data in order to ensure that the learned model satisfies some adversarial goal.

We consider instead an agent who poisons the model updates it sends back to the server.

This attack is a plausible threat in the federated learning setting as the model updates from the agents can (i) directly influence the parameters of the global model via the aggregation algorithm; and (ii) display high variability, due to the non-i.i.d local data at the agents, making it harder to isolate the benign updates from the malicious ones.

Adversary Model: We make the following assumptions regarding the adversary: (i) there is exactly one non-colluding, malicious agent with index m (limited effect of malicious updates on the global model); (ii) the data is distributed among the agents in an i.i.d fashion (making it easier to discriminate between benign and possible malicious updates and harder to achieve attack stealth); (iii) the malicious agent has access to a subset of the training data D m as well as to auxiliary data D aux drawn from the same distribution as the training and test data that are part of its adversarial objective.

Our aim is to explore the possibility of a successful model poisoning attack even for a highly constrained adversary.

A malicious agent can have one of two objectives with regard to the loss and/or classification of a data subset at any time step t in the model poisoning setting: 1.

Increase the overall loss: In this case, the malicious agent wishes to increase the overall loss on a subset D aux = {x i , y i } r i=1 of the data.

The adversarial objective is in this setting is DISPLAYFORM0 DISPLAYFORM1 .

This corresponds to a targeted misclassification attempt by the malicious agent.

In this paper, we will focus on malicious agents trying to attain the second objective, i.e. targeted misclassification.

At first glance, the problem seems like a simple one for the malicious agent to solve.

However, it does not have access to the global parameter vector w t G for the current iteration as is the case in standard poisoning attacks BID15 BID9 and can only influence it though the weight update δ t m it provides to the server S. The simplest formulation of the optimization problem the malicious agent has to solve such that her objective is achieved on the t th iteration is then DISPLAYFORM2

In order to illustrate how our attack strategies work with actual data and models, we use two qualitatively different datasets.

The first is an image dataset, Fashion-MNIST 3 BID21 ) which consists of 28 × 28 grayscale images of clothing and footwear items and has 10 output classes.

The training set contains 60,000 data samples while the test set has 10,000 samples.

We use a Convolutional Neural Network achieving 91.7% accuracy on the test set for the model architecture.

The second dataset is the UCI Adult dataset 4 , which has over 40,000 samples containing information about adults from the 1994 US Census.

The classification problem is to determine if the income for a particular individual is greater (class '0') or less (class '1') than $50, 000 a year.

For this dataset, we use a fully connected neural network achieving 84.8% accuracy on the test set BID6 for the model architecture.

Owing to space constraints, all results for this dataset are in the Appendix.

For both datasets, we study the case with the number of agents k set to 10 and 100.

When k = 10, all the agents are chosen at every iteration, while with k = 100, a tenth of the agents are chosen at random every iteration.

We run federated learning till a pre-specified test accuracy (91% for Fashion MNIST and 84% for the Adult Census data) is reached or the maximum number of time steps have elapsed (40 for k = 10 and 50 for k = 100).

For most of our experiments, we consider the case when r = 1, which implies that the malicious agent aims to misclassify a single example in a desired target class.

For both datasets, a random sample from the test set is chosen as the example to be misclassified.

For the Fashion-MNIST dataset, the sample belongs to class '5' (sandal) with the aim of misclassifying it in class '7' (sneaker) and for the Adult dataset it belongs to class '0' with the aim of misclassifying it in class '1'.

We begin by investigating baseline attacks which do not conform to any notion of stealth.

We then show how simple detection methods at the server may expose the malicious agent and explore the extent to which modifications to the baseline attack can bypass these methods.

In order to solve the exact optimization problem needed to achieve their objective, the malicious agent needs access to the current value of the overall parameter vector w t G , which is inaccessible.

This occurs due to the nature of the federated learning algorithm, where S computes w t G once it has received updates from all agents.

In this case, they have to optimize over an estimate of the value of w t G : DISPLAYFORM0 where f (·) is an estimator forŵ t G based on all the information I t m available to the adversary.

We refer to this as the limited information poisoning objective.

The problem of choosing a good estimator is deferred to Section 4 and the strategies discussed in the remainder of this section make the assumption thatŵ DISPLAYFORM1 In other words, the malicious agent ignores the effects of other agents.

As we shall see, this assumption is often enough to ensure the attack works in practice.

Using the approximation thatŵ DISPLAYFORM0 Depending on the exact structure of the loss, an appropriate optimizer can be chosen.

For our experiments, we will rely on gradient-based optimizers such as SGD which work well for neural networks.

In order to overcome the effect of scaling by α m at the server, the final updateδ t m that is returned, has to be boosted.

Explicit Boosting:

Mimicking a benign agent, the malicious agent can run E m steps of a gradientbased optimizer starting from w Implicit Boosting: While the loss is a function of a weight vector w, we can use the chain rule to obtain the gradient of the loss with respect to the weight update δ, i.e. ∇ δ L = α m ∇ w L.

Then, initializing δ to some appropriate δ ini , the malicious agent can directly minimize with respect to δ.

FIG1 .

The attack is clearly successful at causing the global model to classify the chosen example in the target class.

In fact, after t = 3, the global model is highly confident in its (incorrect) prediction.

The baseline attack using implicit boosting FIG3 ) is much less successful than the explicit boosting baseline, with the adversarial objective only being achieved in 4 of 10 iterations.

Further, it is computationally more expensive, taking an average of 2000 steps to converge at each time step, which is about 4× longer than a benign agent.

Since consistently delayed updates from the malicious agent might lead to it being dropped from the system in practice, we focus on explicit boosting attacks for the remainder of the paper as they do not add as much overhead.

While the baseline attack is successful at meeting the malicious agent's objective, there are methods the server can employ in order to detect if an agent's update is malicious.

We now discuss two possible methods and their implication for the baseline attack.

We note that neither of these methods are part of the standard federated learning algorithm nor do they constitute a full defense at the server.

They are merely metrics that may be utilized in a secure system.

Accuracy checking: When any agent sends a weight update to the server, it can check the validation accuracy of w t i = w t−1 G + δ t i , the model obtained by adding that update to the current state of the global model.

If the resulting model has a validation accuracy much lower than that of the other agents, the server may be able to detect that model as coming from a malicious agent.

This would be particularly effective in the case where the agents have i.i.d.

data.

In FIG1 , the left plot shows the accuracy of the malicious model on the validation data (Acc.

Mal) at each iteration.

This is much lower than the accuracy of the global model (Acc.

Global) and is no better than random for the first few iterations.

Figure 3: Minimum and maximum L 2 distances between weight updates.

For each strategy, we show the spread of L 2 distances between all the benign agents and between the malicious agent and the benign agents.

Going from the baseline attack to the alternating minimization attack with and without distance constraints, we see that the gap in the spread of distances reduces, making the attack stealthier.

The benign agents behave almost identically across strategies, indicating that the malicious agent does not interfere much with their training.

Weight update statistics: There are both qualitative and quantitative methods the server can apply in order to detect weight updates which are malicious, or at the least, different from a majority of the other agents.

We investigate the effectiveness of two such methods.

The first, qualitative method, is the visualization of weight update distributions for each agent.

Since the adversarial objective function is different from the training loss objective used by all the benign agents, we expect the distribution of weight updates to be very different.

This is borne out by the representative weight update distribution at t = 4 observed for the baseline attack in FIG1 .

Compared to the weight update from a benign agent, the update from the malicious agent is much sparser and has a smaller range.

This difference is more pronounced for later time steps (see FIG13 in Appendix B).The second, quantitative method uses the spread of pairwise L p distances between weight update vectors to identify outliers.

At each time step, the server computes the pairwise distances between all the weight updates it receives, and flags those weight updates which are either much closer or much farther away than the others.

In Figure 3 , the spread of L 2 distances between all benign updates and between the malicious update and the benign updates is plotted.

For the baseline attack, both the minimum and maximum distance away from any of the benign updates keeps decreasing over time steps, while it remains relatively constant for the other agents.

This can enable detection of the malicious agent.

To bypass the two detection methods discussed in the previous section, the malicious agent can try to simultaneously optimize over the adversarial objective and training loss for its local data shard Results: In practice, we optimize over batches of D m and concatenate each batch with the single instance {x, τ } to be misclassified, ensuring that the adversarial objective is satisfied.

In fact, as seen in FIG1 in the plot on the right, the adversarial objective is satisfied with high confidence from the first time step t = 1.

Since the entire weight update corresponding to both adversarial and training objectives is boosted, the accuracy of w t m on the validation is low throughout the federated learning process.

Thus, this attack can easily be detected using the accuracy checking method.

Further, while the weight update distribution for this attack FIG1 is visually similar to that of benign agents, its range differs, again enabling detection.

The malicious agent only needs to boost the part of the weight update that corresponds to the adversarial objective.

In the baseline attack, in spite of this being the entire update, the resulting distribution is sparse and of low magnitude compared to a benign agent's updates.

This indicates that the weights update needed to meet the adversarial objective could be hidden in an update that resembled that of a benign agent.

However, as we saw in the previous section, boosting the entire weight update when the training loss is included leads to low validation accuracy.

Further, the concatenation strategy does not allow for parts of the update corresponding to the two different objectives to be decoupled.

To overcome this, we propose an alternating minimization attack strategy which works as follows for iteration t. For each epoch i, the adversarial objective is first minimized starting from w Results: In FIG6 , the plot on the left shows the evolution of the metrics of interest over iterations.

The alternating minimization attack is able to achieve its goals as the accuracy of the malicious model closely matches that of the global model even as the adversarial objective is met with high confidence for all time steps starting from t = 3.

This attack can bypass the accuracy checking method as the accuracy on test data of the malicious model is close to that of the global model.

Qualitatively, the distribution of the malicious weight update FIG6 ) is much more similar to that of the benign weights as compared to the baseline attack.

Further, in Figure 3 , we can see that the spread in distances between the malicious updates and benign updates much closer to that between benign agents compared to the baseline attack.

Thus, this attack is stealthier than the baseline.

To increase the attack stealth, the malicious agent can also add a distance-based constraint onw i,t m , which is the intermediate weight vector generated in the alternating minimization strategy.

There could be multiple local minima which lead to low training loss, but the malicious agent needs to send back a weight update that is as close as possible (in an appropriate distance metric) to the update they would have sent had they been benign.

So, w Constraints based on the empirical distribution of weights such as the Wasserstein or total variation distances may also be used.

The adversarial objective is achieved at the global model with high confidence starting from time step t = 2 and the success of the malicious model on the benign objective closely tracks that of the global model throughout.

The weight update distribution for this attack FIG6 ) is again similar to that of a benign agent.

Further, in Figure 3 , we can see that the distance spread for this attack closely follows and even overlaps that of benign updates throughout, making it hard to detect using the L 2 distance metric.

In this section, we look at how the malicious agent can choose a better estimate for the effect of the other agents' updates at each time step that it is chosen.

In the case when the malicious agent is not chosen at every time step, this estimation is made challenging by the fact that it may not have been chosen for many iterations.

The malicious agent's goal is to choose an appropriate estimate for δ DISPLAYFORM0 Eq. 1.

At a time step t when the malicious agent is chosen, the following information is available to them from the previous time steps they were chosen: i) Global parameter vectors w αm , this will negate the effects of the other agents.

However, due to estimation inaccuracy and the fact that the optimizer has not accounted for this correction, this method leads to poor empirical performance.

Pre-optimization correction: Here, the malicious agent assumes thatŵ DISPLAYFORM1 m .

In other words, the malicious agent optimizes for δ t m assuming it has an accurate estimate of the other agents' updates.

For attacks which use explicit boosting, this involves starting from w

When the malicious agent is chosen at time step t 5 , information regarding the probable updates from the other agents can be obtained from the previous time steps at which the malicious agent was chosen.

Previous step estimate: In this method, the malicious agent's estimateδ t [k]\m assumes that the other agents' cumulative updates were the same at each step since t (the last time step at which at the malicious agent was chosen), i.e.δ DISPLAYFORM0 In the case when the malicious agent is chosen at every time step, this reduces toδ DISPLAYFORM1 Results: Attacks using previous step estimation with the pre-optimization correction are more effective at achieving the adversarial objective for both the baseline and alternating minimization attacks.

In FIG10 , the global model misclassifies the desired sample with a higher confidence for both the baseline and alternating minimization attacks at t = 2.

Neural networks are often treated as black boxes with little transparency into their internal representation or understanding of the underlying basis for their decisions.

Interpretability techniques are designed to alleviate these problems by analyzing various aspects of the network.

These include (i) identifying the relevant features in the input pixel space for a particular decision via Layerwise Relevance Propagation (LRP) techniques BID14 ); (ii) visualizing the association between neuron activations and image features (Guided Backprop (Springenberg et al. (2014) ), DeConvNet (Zeiler & Fergus (2014) )); (iii) using gradients for attributing prediction scores to input features (e.g., Integrated Gradients BID20 ), or generating sensitivity and saliency maps (SmoothGrad BID18 ), Gradient Saliency Maps BID17 )) and so on.

The semantic relevance of the generated visualization, relative to the input, is then used to explain the model decision.

These interpretability techniques, in many ways, provide insights into the internal feature representations and working of a neural network.

Therefore, we used a suite of these techniques to try and discriminate between the behavior of a benign global model and one that has been trained to satisfy the adversarial objective of misclassifying a single example.

FIG11 compares the output of the various techniques for both the benign and malicious models on a random auxiliary data sample.

Targeted perturbation of the model parameters coupled with tightly bounded noise ensures that the internal representations, and relevant input features used by the two models, for the same input, are almost visually imperceptible.

This reinforces the stealth achieved by our attacks along with respect to another measure of stealth, namely various interpretability-based detection techniques.

In this paper, we have started an exploration of the vulnerability of multi-party machine learning algorithms such as federated learning to model poisoning adversaries, who can take advantage of the very privacy these models are designed to provide.

In future work, we plan to explore more sophisticated detection strategies at the server, which can provide guarantees against the type of attacker we have considered here.

In particular, notions of distances between weight distributions are promising defensive tools.

Our attacks in this paper demonstrate that federated learning in its basic form is very vulnerable to model poisoning adversaries, as are recently proposed Byzantine resilient aggregation mechanisms.

While detection mechanisms can make these attacks more challenging, they can be overcome, demonstrating that multi-party machine learning algorithms robust to attackers of the type considered here must be developed.

When the number of agents increases to k = 100, the malicious agent is not selected in every step.

Further, the size of |D m | decreases, which makes the benign training step in the alternating minimization attack more challenging.

The challenges posed in this setting are reflected in Figure 8 , where although the baseline attack is able to introduce a targeted backdoor, it cannot ensure it for every step due to steps where only benign agents provide updates.

The alternating minimization attack is also able to introduce the backdoor, as well as increase the classification accuracy of the malicious model on test data.

However, the improvement in performance is limited by the paucity of data for the malicious agent.

It is an open question if data augmentation could help improve this accuracy.

Figure B shows the evolution of weight update distributions for the 4 different attack strategies on the CNN trained on the Faishon MNIST dataset.

Time slices of this evolution were shown in the main text of the paper.

The baseline and concatenated training attacks lead to weight update distributions that differ widely for benign and malicious agents.

The alternating minimization attack without distance constraints reduces this qualitative difference somewhat but the closest weight update distributions are obtained with the alternating minimization attack with distance constraints.

C BYPASSING BYZANTINE-RESILIENT AGGREGATION MECHANISMS BID2 recently proposed a gradient aggregation mechanism known as 'Krum' that is provably resilient to Byzantine adversaries.

We choose to evaluate Krum as it is efficient, provably resilient and can be used a building block for better mechanisms BID13 .

As stated in the introduction, the aim of Byzantine adversaries considered in this work and others BID5 BID13 ; ; BID23 ) is to ensure convergence to ineffective models.

The goals of the adversary in this paper are to ensure convergence to effective models with targeted backdoors.

This difference in objectives leads to 'Krum' being ineffective against our attacks.

We now briefly describe Krum.

Given n agents of which f are Byzantine, Krum requires that n ≥ 2f + 3.

At any time step t, updates (δ t 1 , . . .

, δ t n ) are received at the server.

For each δ t i , the n − f − 2 closest (in terms of L p norm) other updates are chosen to form a set C i and their distances added up to give a score S(δ In FIG1 , we see the effect of our attack strategies on Krum with a boosting factor of λ = 2 for a federated learning setup with 10 agents.

Since there is no need to overcome the constant scaling factor α m , the attacks can use a much smaller boosting factor λ to ensure the global model has the targeted backdoor.

Even with the baseline attack, the malicious agent's update is the one chosen by Krum for 34 of 40 time steps but the global model is unable to attain high test accuracy.

The alternating minimization attack ensures that the global model maintains relatively high test accuracy while the malicious agent is chosen for 26 of 40 time steps.

These results conclusively demonstrate the effectiveness of model poisoning attacks against Krum.

<|TLDR|>

@highlight

Effective model poisoning attacks on federated learning able to cause high-confidence targeted misclassification of desired inputs