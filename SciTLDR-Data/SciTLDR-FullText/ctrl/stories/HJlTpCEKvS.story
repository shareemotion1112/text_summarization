Many computer vision applications require solving multiple tasks in real-time.

A neural network can be trained to solve multiple tasks simultaneously using 'multi-task learning'.

This saves computation at inference time as only a single network needs to be evaluated.

Unfortunately, this often leads to inferior overall performance as task objectives compete, which consequently poses the question: which tasks should and should not be learned together in one network when employing multi-task learning?

We systematically study task cooperation and competition and propose a framework for assigning tasks to a few neural networks such that cooperating tasks are computed by the same neural network, while competing tasks are computed by different networks.

Our framework offers a time-accuracy trade-off and can produce better accuracy using less inference time than not only a single large multi-task neural network but also many single-task networks.

Many applications, especially robotics and autonomous vehicles, are chiefly interested in using multi-task learning to reduce the inference time and computational complexity required to estimate many characteristics of visual input.

For example, an autonomous vehicle may need to detect the location of pedestrians, determine a per-pixel depth, and predict objects' trajectories, all within tens of milliseconds.

In multi-task learning, multiple tasks are solved at the same time, typically with a single neural network.

In addition to reduced inference time, solving a set of tasks jointly rather than independently can, in theory, have other benefits such as improved prediction accuracy, increased data efficiency, and reduced training time.

Unfortunately, the quality of predictions are often observed to suffer when a network is tasked with making multiple predictions.

This is because learning objectives can have complex and unknown dynamics and may compete.

In fact, multi-task performance can suffer so much that smaller independent networks are often superior (as we will see in the experiments section).

We refer to any situation in which the competing priorities of the network cause poor task performance as crosstalk.

On the other hand, when task objectives do not interfere much with each other, performance on both tasks can be maintained or even improved when jointly trained.

Intuitively, this loss or gain of quality seems to depend on the relationship between the jointly trained tasks.

Prior work has studied the relationship between tasks for transfer learning (Zamir et al. (2018) ).

However, we find that transfer relationships are not highly predictive of multi-task relationships.

In addition to studying multi-task relationships, we attempt to determine how to produce good prediction accuracy under a limited inference time budget by assigning competing tasks to separate networks and cooperating tasks to the same network.

More concretely, this leads to the following problem: Given a set of tasks, T , and a computational budget b (e.g., maximum allowable inference time), what is the optimal way to assign tasks to networks with combined cost ≤ b such that a combined measure of task performances is maximized?

To this end, we develop a computational framework for choosing the best tasks to group together in order to have a small number of separate deep neural networks that completely cover the task set and that maximize task performance under a given computational budget.

We make the intriguing Figure 1 : Given five tasks to solve, there are many ways that they can be split into task groups for multitask learning.

How do we find the best one?

We propose a computational framework that, for instance, suggests the following grouping to achieve the lowest total loss, using a computational budget of 2.5 units: train network A to solve Semantic Segmentation, Depth Estimation, and Surface Normal Prediction; train network B to solve Keypoint Detection, Edge Detection, and Surface Normal Prediction; train network C with a less computationally expensive encoder to solve Surface Normal Prediction alone; including Surface Normals as an output in the first two networks were found advantageous for improving the other outputs, while the best Normals were predicted by the third network.

This task grouping outperforms all other feasible ones, including learning all five tasks in one large network or using five dedicated smaller networks.

observation that the inclusion of an additional task in a network can potentially improve the accuracy of the other tasks, even though the performance of the added task might be poor.

This can be viewed as regularizing or guiding the loss of one task by adding an additional loss, as often employed in curriculum learning or network regularization Bengio et al. (2009) .

Achieving this, of course, depends on picking the proper regularizing task -our system can take advantage of this phenomenon, as schematically shown in Figure 1 .

This paper has two main contributions.

In Section 3, we outline a framework for systematically assigning tasks to networks in order to achieve the best total prediction accuracy with a limited inference-time budget.

We then analyze the resulting accuracy and show that selecting the best assignment of tasks to groups is critical for good performance.

Secondly, in Section 6, we analyze situations in which multi-task learning helps and when it doesn't, quantify the compatibilities of various task combinations for multi-task learning, compare them to the transfer learning task affinities, and discuss the implications.

Moreover, we analyze the factors that influence multi-task affinities.

Multi-Task Learning: See Ruder (2017) for a good overview of multi-task learning.

The authors identify two clusters of contemporary techniques that we believe cover the space well, hard parameter sharing and soft parameter sharing.

In brief, the primary difference between the majority of the existing works and our study is that we wish to understand the relationships between tasks and find compatible groupings of tasks for any given set of tasks, rather than designing a neural network architecture to solve a particular fixed set of tasks well.

A known contemporary example of hard parameter sharing in computer vision is UberNet (Kokkinos (2017) ).

The authors tackle 7 computer vision problems using hard parameter sharing.

The authors focus on reducing the computational cost of training for hard parameter sharing, but experience a rapid degradation in performance as more tasks are added to the network.

Hard parameter sharing is also used in many other works such as (Thrun (1996); Caruana (1997); Nekrasov et al. (2018) ; Dvornik et al. (2017) ; Kendall et al. (2018) ; Bilen & Vedaldi (2016) ; Pentina & Lampert (2017) ; Doersch & Zisserman (2017); Zamir et al. (2016); Long et al. (2017); Mercier et al. (2018) ; d. Miranda et al. (2012) ; Zhou et al. (2018) ; Rudd et al. (2016) ).

Other works, such as (Sener & Koltun (2018) ) and (Chen et al. (2018b) ), aim to dynamically reweight each task's loss during training.

The former work finds weights that provably lead to a Pareto-optimal solution, while the latter attempts to find weights that balance the influence of each task on network weights.

Finally, (Bingel & Søgaard (2017) ) studies task interaction for NLP.

In soft or partial parameter sharing, either there is a separate set of parameters per task, or a significant fraction of the parameters are unshared.

The models are tied together either by information sharing or by requiring parameters to be similar.

Examples include (Dai et al. (2016) ; Duong et al. (2015) ; Misra et al. (2016) ; Tessler et al. (2017) ; Yang & Hospedales (2017) ; Lu et al. (2017) ).

The canonical example of soft parameter sharing can be seen in (Duong et al. (2015) ).

The authors are interested in designing a deep dependency parser for languages such as Irish that do not have much treebank data available.

They tie the weights of two networks together by adding an L2 distance penalty between corresponding weights and show substantial improvement.

Another example of soft parameter sharing is Cross-stitch Networks (Misra et al. (2016) ).

Starting with separate networks for two tasks, the authors add 'cross-stitch units' between them, which allow each network to peek at the other network's hidden layers.

This approach reduces but does not eliminate task interfearence, and the overall performance is less sensitive to the relative loss weights.

Unlike our method, none of the aforementioned works attempt to discover good groups of tasks to train together.

Also, soft parameter sharing does not reduce inference time, a major goal of ours.

Transfer Learning: Transfer learning (Pratt (1993) Rusu et al. (2016) ) is similar to multi-task learning in that solutions are learned for multiple tasks.

Unlike multi-task learning, however, transfer learning methods often assume that a model for a source task is given and then adapt that model to a target task.

Transfer learning methods generally neither seek any benefit for source tasks nor a reduction in inference time as their main objective.

Neural Architecture Search (NAS): Many recent works search the space of deep learning architectures to find ones that perform well (Zoph & Le, 2017; Pham et al., 2018; Xie et al., 2019; Elsken et al., 2019; Zhou et al., 2019; Baker et al., 2017; Real et al., 2018) .

This is related to our work as we search the space of task groupings.

Just as with NAS, the found task groupings often perform better than human-engineered ones.

Task Relationships:

Our work is most related to Taskonomy (Zamir et al. (2018)), where the authors studied the relationships between visual tasks for transfer learning and introduced a dataset with over 4 million images and corresponding labels for 26 tasks.

This was followed by a number of recent works, which further analyzed task relationships (Pal & Balasubramanian (2019) ; Dwivedi & Roig. (2019) ; Achille et al. (2019) ; Wang et al. (2019) ) for transfer learning.

While they extract relationships between these tasks for transfer learning, we are interested in the multi-task learning setting.

Interestingly, we find notable differences between transfer task affinity and multi-task affinity.

Their method also differs in that they are interested in labeled-data efficiency and not inference-time efficiency.

Finally, the transfer quantification approach taken by Taskonomy (readout functions) is only capable of finding relationships between the high-level bottleneck representations developed for each task, whereas structural similarities between tasks at all levels are potentially relevant for multi-task learning.

Our goal is to find an assignment of tasks to networks that results in the best overall loss.

Our strategy is to select from a large set of candidate networks to include in our final solution.

We define the problem as follows: We want to minimize the overall loss on a set of tasks T = {t 1 , t 2 , ..., t k } given a limited inference time budget, b, which is the total amount of time we have to complete all tasks.

Each neural network that solves some subset of T and that could potentially be a part of the final solution is denoted by n.

It has an associated inference time cost, c n , and a loss for each task, L(n, t i ) (which is ∞ for each task the network does not attempt to solve).

A solution S is a set of networks that together solve all tasks.

The computational cost of a solution is cost(S) = n∈S c n .

The loss of a solution on a task, L(S, t i ), is the lowest loss on that task among the solution's networks 1 , L(S,

We want to find the solution with the lowest overall loss and a cost that is under our budget, S b = argmin S:cost(S)≤b L(S).

For a given task set T , we wish to determine not just how well each pair of tasks performs when trained together, but also how well each combination of tasks performs together so that we can capture higher-order task relationships.

To that end, our candidate set of networks contains all 2 |T | − 1 possible groupings:

|T | 1 networks with one task,

|T | 2 networks with two tasks,

networks with three tasks, etc.

For the five tasks we use in our experiments, this is 31 networks, of which five are single-task networks.

The size of the networks is another design choice, and to somewhat explore its effects we also include 5 single task networks each with half of the computational cost of a standard network.

This brings our total up to 36 networks.

Consider the situation in which we have an initial candidate set C 0 = {n 1 , n 2 , ..., n m } of fullytrained networks that each solve some subset of our task set T .

Our goal is to choose a subset of C 0 that solve all the tasks with total inference time under budget b and the lowest overall loss.

More formally, we want to find a solution

It can be shown that solving this problem is NP-hard in general (reduction from SET-COVER).

However, many techniques exist that can optimally solve most reasonably-sized instances of problems like these in acceptable amounts of time.

All of these techniques produce the same solutions.

We chose to use a branch-and-bound-like algorithm for finding our optimal solutions (shown as Algorithm 1 in the Appendix), but in principle the exact same solutions could be achieved by other optimization methods, such as encoding the problem as a binary integer program (BIP) and solving it in a way similar to Taskonomy (Zamir et al. (2018)).

Most contemporary MTL works use fewer than 4 unique task types, but in principal, the NP-hard nature of the optimization problem does limit the number of candidate solutions that can be considered.

However, using synthetic inputs, we found that our branch-and-bound like approach requires less time than network training for all 2 |T | − 1 + |T | candidates for fewer than ten tasks.

Scaling beyond that would require approximations or stronger optimization techniques.

This section describes two techniques for reducing the training time required to obtain a collection of networks as input to the network selection algorithm.

Our goal is to produce task groupings with results similar to the ones produced by the complete search, but with less training time burden.

Both techniques involve predicting the performance of a network without actually training it to convergence.

The first technique involves training each of the networks for a short amount of time, and the second involves inferring how networks trained on more than two tasks will perform based on how networks trained on two tasks perform.

We found a moderately high correlation (Pearson's r = 0.49) between the validation loss of our neural networks after a pass through just 20% of our data and the final test loss of the fully trained networks.

This implies that the task relationship trends stabilize early.

We fine that we can get decent results by running network selection on the lightly trained networks, and then simply training the chosen networks to convergence.

For our setup, this technique reduces the training time burden by about 20x over fully training all candiate networks and would require fewer than 150 GPU hours to execute.

This is only 35% training-time overhead.

Obviously, this technique does come with a prediction accuracy penalty.

Because the correlation between early network performance and final network performance is not perfect, the decisions made by network selection are no longer guaranteed to be optimal once networks are trained to convergence.

We call this approximation the Early Stopping Approximation (ESA) and present the results of using this technique in Section 5.

Do the performances of a network trained with tasks A and B, another trained with tasks A and C, and a third trained with tasks B and C tell us anything about the performance of a network trained on tasks A, B, and C?

As it turns out, the answer is yes.

Although this ignores complex task interactions and nonlinearities, a simple average of the first-order networks' accuracies was a good indicator of the accuracy of a higher-order network.

Experimentally, this prediction strategy has an average max ratio error of only 5.2% on our candidate networks.

Using this strategy, we can predict the performance of all networks with three or more tasks using the performance of all of the fully trained two task networks.

First, simply train all networks with two or fewer tasks to convergence.

Then predict the performance of higher-order networks.

Finally, run network selection on both groups.

With our setup (see Section 4), this strategy saves training time by only about 50%, compared with 95% for the early stopping approximation, and it still comes with a prediction quality penalty.

However, this technique requires only a quadratic number of networks to be trained rather than an exponential number, and would therefore win out when the number of tasks is large.

We call this strategy the Higher Order Approximation (HOA), and present its results in Section 5.

We perform our evaluation using the Taskonomy dataset (Zamir et al. (2018)), which is currently the largest multi-task dataset in vision with diverse tasks.

The data was obtained from 3D scans of about 600 buildings.

There are 4,076,375 examples, which we divided into 3,974,199 training instances, 52,000 validation instances, and 50,176 test instances.

There was no overlap in the buildings that appeared in the training and test sets.

All data labels were normalized (x = 0, σ = 1).

Our framework is agnostic to the particular set of tasks.

We have chosen to perform the study using five tasks in Taskonomy: Semantic Segmentation, Depth Estimation, Surface Normal Prediction, Keypoint Detection, and Edge Detection, so that one semantic task, two 3D tasks, and two 2D tasks are included.

These tasks were chosen to be representative of major task categories, but also to have enough overlap in order to test the hypothesis that similar tasks will train well together.

Crossentropy loss was used for Semantic Segmentation, while an L1 loss was used for all other tasks.

Network Architecture:

The proposed framework can work with any network architecture.

In our experiments, all of the networks used a standard encoder-decoder architecture with a modified Xception (Chollet (2017) ) encoder.

Our choice of architecture is not critical and was chosen for reasonably fast inference time performance.

The Xception network encoder was simplified to have 17 layers and the middle flow layers were reduced to having 512 rather than 728 channels.

All maxpooling layers were replaced by 2 × 2 convolution layers with a stride of 2 (similar to Chen et al. (2018a) ).

The full-size encoder had about 4 million parameters.

All networks had an input image size of 256x256.

We measure inference time in units of the time taken to do inference for one of our full-size encoders.

We call this a Standard Network Time (SNT).

This corresponds to 2.28 billion multiply-adds and about 4 ms/image on a single Nvidia RTX 2080 Ti.

Our decoders were designed to be lightweight and have four transposed convolutional layers (Noh et al. (2015) ) and four separable convolutional layers (Chollet (2017) ).

Every decoder has about 116,000 parameters.

All training was done using PyTorch (Paszke et al. (2017) ) with Apex for fp16 acceleration (Micikevicius et al. (2017) ).

Trained Networks: As described in Section 3.1, we trained 31 networks with full sized encoders and standard decoders.

26 were multi-task networks and 5 were single task networks.

Another five single-task networks were trained, each having a half-size encoder and a standard decoder.

These 36 networks were included in network optimization as C 0 .

20 smaller, single-task networks of various sizes were also trained to be used in the baselines and the analysis of Section 6, but not used for network selection.

In order to produce our smaller models, we shrunk the number of channels in every layer of the encoder such that it had the appropriate number of parameters and flops.

The training loss we used was the unweighted mean of the losses for the included tasks.

Networks were trained with an initial learning rate of 0.2, which was reduced by half every time the training loss stopped decreasing.

Networks were trained until their validation loss stopped improving, typically requiring only 4-8 passes through the dataset.

The network with the highest validation loss (checked after each epoch of 20% of our data) was saved.

The performance scores used for network selection were calculated on the validation set.

We computed solutions for inference time budgets from 1 to 5 at increments of 0.5.

Each solution chosen was evaluated on the test set.

We compare our results with conventional methods, such as five single-task networks and a single network with all tasks trained jointly.

We also compare with two multi-task methods in the literature.

The first one is Sener & Koltun (2018) .

We found that their algorithm under-weighted the Semantic Segmentation task too aggressively, leading to poor performance on the task and poor performance overall compared to a simple sum of task losses.

We speculate that this is because semantic segmentation's loss behaves differently from the other losses.

Next we compared to GradNorm (Chen et al. (2018b) ).

The results here were also slightly worse than classical MTL with uniform task weights.

In any event, these techniques are orthogonal to ours and can be used in conjunction for situations in which they lead to better solutions than simply summing losses.

Finally, we compare our results to two control baselines illustrative of the importance of making good choices about which tasks to train together, 'Random' and 'Pessimal.' 'Random' is a solution consisting of valid random task groupings that solve our five tasks.

The reported values are the average of a thousand random trials. '

Pessimal' is a solution in which we choose the networks that lead to the worst overall performance, though the solution's performance on each task is still the best among its networks.

Each baseline was evaluated with multiple encoder sizes so that all models' results could be compared at many inference time budgets.

Figure 2 shows the task groups that were chosen for each technique, and Figure 3 shows the performance of these groups along with those of our baselines.

We can see that each of our methods outperforms our traditional baselines for every computational budget.

When the computational budget is only 1 SNT, all of our methods must select the same model-a traditional multi-task network with a 1 SNT encoder and five decoders.

This strategy outperforms GradNorm, Sener & Koltun (2018) , and individual training.

However, solutions that utilize multiple networks outperform this traditional strategy for every budget > 1.5-better performance can always be achieved by grouping tasks according to their compatibility.

Table 7 .

When the computational budget is effectively unlimited (5 SNT), our optimal method picks five networks, each of which is used to make predictions for a separate task.

However, three of the networks are trained with three tasks each, while only two are trained with one task each.

This shows that the representations learned through multi-task learning were found to be best for three of our tasks (s, d, and e), whereas two of our tasks (n and k) are best solved individually.

We also see that our optimal technique using 2.5 SNT and our Higher Order Approximation using 3.5 SNT can both outperform five individual networks (which uses 5 SNT).

Total Loss All-in-one (triple-size resnet18) 0.50925 Five Individual (resnet18s .6-size each) 0.53484 nKE, SDn, N (3 standard resnet18's) 0.50658 In order to determine how these task groupings generalize to other architectures, we retrained our best solution for 3 SNT using resnet18 ).

The results in Table 1 suggest that good task groupings for one architecture are likely to be good in another, though to a lesser extent.

Task affinities seem to be somewhat architecture-dependent, so for the very best results, task selection must be run for each architecture choice.

Figure 4 allows qualitative comparison between our methods and our baselines.

We can see clear visual issues with each of our baselines that are not present in our methods.

Both of our approximate methods produce predictions similar to the optimal task grouping.

The data generated by the above evaluation presents an opportunity to analyze how tasks interact in a multi-task setting, and allows us to compare with some of the vast body of research in transfer learning, such as Taskonomy (Zamir et al. (2018) Table 4 : The transfer learning affinities between pairs of tasks according to the authors of Taskonomy (Zamir et al. (2018)).

Forward and backward transfer affinities are averaged.

Transfer Affinity Multi-Task Affinity In order to determine the between task affinity for multi-task learning, we took the average of our first-order relationships matrix (Table 2) and its transpose.

The result is shown in Table 3.

The pair with the highest affinity by this metric are Surface Normal Prediction and 2D Edge Detection.

Our two 3D tasks, Depth Estimation and Surface Normal Prediction, do not score highly on this similarity metric.

This contrasts with the findings for transfer learning in Taskonomy (Table 4) , in which they have the highest affinity.

Our two 2D tasks also do not score highly.

We speculate that the Normals task naturally preserves edges, while Depth and Normals (for example) don't add much training signal to each other.

See Section A.3 for more on factors that influence multi-task affinity.

Figure 5 depicts the relationship between transfer learning affinities and multi-task affinities, which surprisingly seem to be negatively correlated in our high-data scenario.

This suggests that it might be better to train dissimilar tasks together.

This could be because dissimilar tasks are able to provide stronger and more meaningful regularization.

More research is necessary to discover when and if this correlation and explanation hold.

We describe the problem of task compatibility as it pertains to multi-task learning.

We provide an algorithm and computational framework for determining which tasks should be trained jointly and which tasks should be trained separately.

Our solution can take advantage of situations in which joint training is beneficial to some tasks but not others in the same group.

For many use cases, this framework is sufficient, but it can be costly at training time.

Hence, we offer two strategies for coping with this issue and evaluate their performance.

Our methods outperform single-task networks, a multi-task network with all tasks trained jointly, as well as other baselines.

Finally, we use this opportunity to analyze how particular tasks interact in a multi-task setting and compare that with previous results on transfer learning task interactions.

A.1 NETWORK SELECTION ALGORITHM

Input: C r , a running set of candidate networks, each with an associated cost c ∈ R and a performance score for each task the network solves.

Initially, C r = C 0 Input: S r ⊆ C 0 , a running solution, initially Ø Input: b r ∈ R, the remaining time budget, initially b

Most promising networks first 4:

Best ←

S r

for n ∈ C r do 6:

C r ← C r \ n \ is set subtraction.

S i ← S r ∪ {n} 8:

Child ← GETBESTNETWORKS(C r , S i , b i )

Best ← BETTER(Best, Child)

return Best 12: function FILTER(C r , S r , b r )

Remove networks from C r with c n > b r .

14:

Remove networks from C r that cannot improve S r 's performance on any task.

return S 2 Algorithm 1 chooses the best subset of networks in our collection, subject to the inference time budget constraint.

The algorithm recursively explores the space of solutions and prunes branches that cannot lead to optimal solutions.

The recursion terminates when the budget is exhausted, at which point C r becomes empty and the loop body does not execute.

The sorting step on line 3 requires a heuristic upon which to sort.

We found that ranking models based on how much they improve the current solution, S, works well.

It should be noted that this algorithm always produces an optimal solution, regardless of which sorting heuristic is used.

However, better sorting heuristics reduce the running time because subsequent iterations will more readily detect and prune portions of the search space that cannot contain an optimal solution.

In our setup, we tried variants of problems with 5 tasks and 36 networks, and all of them took less than a second to solve.

The definition of the BETTER() function is application-specific.

For our experiments, we prefer networks that have the lowest total loss across all five tasks.

Other applications may have hard performance requirements for some of the tasks, and performance on one of these tasks cannot be sacrificed in order to achieve better performance on another task.

Such application-specific constraints can be encoded in BETTER().

In order to determine how well network selection works for different task sets, we re-ran network selection on all five 4-task subsets of our task set.

The performance average of all 5 sets is shown in Figure 6 .

We see that our techniques generalize at least to subsets of our studied tasks.

The finding that Depth and Normals don't cooperate is counter to much of the multitask learning literature such as Wang et al. (2016) , Qi et al. (2018) , and Zhang et al. (2019) .

However, the majority of these works use training sets with fewer than 100k instances, while we use nearly 4 million training instances.

Table 5 shows the loss obtained on our setup when we limit to only 100k training instances.

The fact that task affinities can change depending on the amount of available training data demonstrates the necessity of using an empirical approach like ours for finding task affinities and groupings.

A. Table 9 : The test set performance of our 31 networks on each task that they solve.

<|TLDR|>

@highlight

We analyze what tasks are best learned together in one network, and which are best to learn separately. 