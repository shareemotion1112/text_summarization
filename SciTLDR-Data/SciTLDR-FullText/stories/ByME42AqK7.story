Architecture search aims at automatically finding neural architectures that are competitive with architectures designed by human experts.

While recent approaches have achieved state-of-the-art predictive performance for image recognition, they are problematic under resource constraints for two reasons: (1) the neural architectures found are solely optimized for high predictive performance, without penalizing excessive resource consumption; (2)most architecture search methods require vast computational resources.

We address the first shortcoming by proposing LEMONADE, an evolutionary algorithm for multi-objective architecture search that allows approximating the Pareto-front of architectures under multiple objectives, such as predictive performance and number of parameters, in a single run of the method.

We address the second shortcoming by proposing a Lamarckian inheritance mechanism for LEMONADE which generates children networks that are warmstarted with the predictive performance of their trained parents.

This is accomplished by using (approximate) network morphism operators for generating children.

The combination of these two contributions allows finding models that are on par or even outperform different-sized NASNets, MobileNets, MobileNets V2 and Wide Residual Networks on CIFAR-10 and ImageNet64x64 within only one week on eight GPUs, which is about 20-40x less compute power than previous architecture search methods that yield state-of-the-art performance.

Deep learning has enabled remarkable progress on a variety of perceptual tasks, such as image recognition BID12 , speech recognition , and machine translation BID0 .

One crucial aspect for this progress are novel neural architectures BID25 He et al., 2016; BID7 .

Currently employed architectures have mostly been developed manually by human experts, which is a time-consuming and error-prone process.

Because of this, there is growing interest in automatic architecture search methods (Elsken et al., 2018) .

Some of the architectures found in an automated way have already outperformed the best manually-designed ones; however, algorithms such as by BID32 ; ; BID20 BID36 for finding these architectures require enormous computational resources often in the range of thousands of GPU days.

Prior work on architecture search has typically framed the problem as a single-objective optimization problem.

However, most applications of deep learning do not only require high predictive performance on unseen data but also low resource-consumption in terms of, e.g., inference time, model size or energy consumption.

Moreover, there is typically an implicit trade-off between predictive performance and consumption of resources.

Recently, several architectures have been manually designed that aim at reducing resource-consumption while retaining high predictive performance BID8 BID22 .

Automatically found neural architectures have also been down-scaled to reduce resource consumption .

However, very little previous work has taken the trade-off between resource-consumption and predictive performance into account during automatic architecture search.

In this work, we make the following two main contributions:1.

To overcome the need for thousands of GPU days BID32 BID21 , we make use of operators acting on the space of neural network architectures that preserve the function a network represents, dubbed network morphisms (Chen et al., 2015; BID27 , obviating training from scratch and thereby substantially reducing the required training time per network.

This mechanism can be interpreted as Lamarckian inheritance in the context of evolutionary algorithms, where Lamarckism refers to a mechanism which allows passing skills acquired during an individual's lifetime (e.g., by means of learning), on to children by means of inheritance.

Since network morphisms are limited to solely increasing a network's size (and therefore likely also resource consumption), we introduce approximate network morphisms (Section 3.2) to also allow shrinking networks, which is essential in the context of multi-objective search.

The proposed Lamarckian inheritance mechanism could in principle be combined with any evolutionary algorithm for architecture search, or any other method using (a combination of) localized changes in architecture space.2.

We propose a Lamarckian Evolutionary algorithm for Multi-Objective Neural Architecture DEsign, dubbed LEMONADE, Section 4, which is suited for the joint optimization of several objectives, such as predictive performance, inference time, or number of parameters.

LEMONADE maintains a population of networks on an approximation of the Pareto front of the multiple objectives.

In contrast to generic multi-objective algorithms, LEMONADE exploits that evaluating certain objectives (such as an architecture's number of parameters) is cheap while evaluating the predictive performance on validation data is expensive (since it requires training the model first).

Thus, LEMONADE handles its various objectives differently: it first selects a subset of architectures, assigning higher probability to architectures that would fill gaps on the Pareto front for the "cheap" objectives; then, it trains and evaluates only this subset, further reducing the computational resource requirements during architecture search.

In contrast to other multi-objective architecture search methods, LEMONADE (i) does not require to define a trade-off between performance and other objectives a-priori (e.g., by weighting objectives when using scalarization methods) but rather returns a set of architectures, which allows the user to select a suitable model a-posteriori; (ii) LEMONADE does not require to be initialized with well performing architectures; it can be initialized with trivial architectures and hence requires less prior knowledge.

Also, LEMONADE can handle various search spaces, including complex topologies with multiple branches and skip connections.

We evaluate LEMONADE for up to five objectives on two different search spaces for image classification: (i) non-modularized architectures and (ii) cells that are used as repeatable building blocks within an architecture BID31 and also allow transfer to other data sets.

LEMONADE returns a population of CNNs covering architectures with 10 000 to 10 000 000 parameters.

Within only 5 days on 16 GPUs, LEMONADE discovers architectures that are competitive in terms of predictive performance and resource consumption with hand-designed networks, such as MobileNet V2 BID22 , as well as architectures that were automatically designed using 40x greater resources and other multi-objective methods (Dong et al., 2018) .

Multi-objective Optimization Multi-objective optimization BID17 ) deals with problems that have multiple, complementary objective functions f 1 , . . .

, f n .

Let N be the space of feasible solutions N (in our case the space of feasible neural architectures).

In general, multi-objective optimization deals with finding N * ∈ N that minimizes the objectives f 1 , . . .

, f n .

However, typically there is no single N * that minimizes all objectives at the same time.

In contrast, there are multiple Pareto-optimal solutions that are optimal in the sense that one cannot reduce any f i without increasing at least one f j .

More formally, a solution N Pareto-dominates another solution DISPLAYFORM0 ).

The Pareto-optimal solutions N was recently proposed to frame NAS as a reinforcement learning (RL) problem, where the reward of the RL agent is based on the validation performance of the trained architecture BID1 BID32 BID31 BID19 .

BID32 use a recurrent neural network to generate a string representing the neural architecture.

In a follow-up work, search for cells, which are repeated according to a fixed macro architecture to generate the eventual architecture.

Defining the architecture based on a cell simplifies the search space.

An alternative to using RL are neuro-evolutionary approaches that use genetic algorithms for optimizing the neural architecture BID24 BID14 BID21 BID18 BID28 .

In contrast to these works, our proposed method is applicable for multi-objective optimization and employs Lamarckian inheritance, i.e, learned parameters are passed on to a network's offspring.

A related approach to our Lamarckian evolution is population-based training BID9 , which, however, focuses on hyperparameter optimization and not on the specific properties of the optimization of neural architectures.

We note that it would be possible to also include the evolution of hyperparameters in our work.

Unfortunately, most of the aforementioned approaches require vast computational resources since they need to train and validate thousands of neural architectures; e.g., BID32 trained over 10.000 neural architectures, requiring thousands of GPU days.

One way of speeding up evaluation is to predict performance of a (partially) trained model (Domhan et al., 2015; Baker et al., 2017b; BID11 BID13 .

Works on performance prediction are complementary to our work and could be incorporated in the future.

One-Shot Architecture Search is another promising approach for speeding up performance estimation, which treats all architectures as different subgraphs of a supergraph (the one-shot model) and shares weights between architectures BID23 Brock et al., 2017; BID19 BID15 Bender et al., 2018) .

Only the weights of a single one-shot model need to be trained, and architectures (which are just subgraphs of the one-shot model) can then be evaluated without any separate training.

However, a general limitation of one-shot NAS is that the supergraph defined a-priori restricts the search space to its subgraphs.

Moreover, approaches which require that the entire supergraph resides in GPU memory during architecture search will be restricted to relatively small supergraphs.

It is also not obvious how one-shot models could be employed for multi-objective optimization as all subgraphs of the one-shot models are of roughly the same size and it is not clear if weight sharing would work for very different-sized architectures.

LEMONADE does not suffer from any of these disadvantages; it can handle arbitrary large, unconstrained search spaces while still being efficient.

Elsken et al. (2017); Cai et al. (2018a) proposed to employ the concept of network morphisms (see Section 3.1).

The basic idea is to initialize weights of newly generated neural architectures based on weights of similar, already trained architectures so that they have the same accuracy.

This pretrained initialization allows reducing the large cost of training all architectures from scratch.

Our work extends this approach by introducing approximate network morphisms, making the use of such operators suitable for multi-objective optimization.

Multi-objective Neural Architecture Search Very recently, there has also been some work on multi-objective neural architecture search BID10 Dong et al., 2018; BID26 with the goal of not solely optimizing the accuracy on a given task but also considering resource consumption.

BID10 parameterize an architecture by a fixed-length vector description, which limits the architecture search space drastically.

In parallel, independent work to ours, Dong et al. (2018) extend PNAS BID13 by considering multiple objective during the model selection step.

However, they employ CondenseNet BID6 as a base network and solely optimize building blocks within the network which makes the search less interesting as (i) the base network is by default already well performing and (ii) the search space is again limited.

BID26 use a weighted product method (Deb & Kalyanmoy, 2001 ) to obtain a scalarized objective.

However, this scalarization comes with the drawback of weighting the objectives a-priori, which might not be suitable for certain applications.

In contrast to all mentioned work, LEMONADE (i) does not require a complex macro architecture but rather can start from trivial initial networks, (ii) can handle arbitrary search spaces, (iii) does not require to define hard constraints or weights on objectives a-priori.

Let N (X ) denote a space of neural networks, where each element N ∈ N (X ) is a mapping from X ⊂ R n to some other space, e.g., mapping images to labels.

A network operator T : DISPLAYFORM0 We now discuss two specific classes of network operators, namely network morphisms and approximate network morphisms.

Operators from these two classes will later on serve as mutations in our evolutionary algorithm.

Chen et al. FORMULA6 introduced two function-preserving operators for deepening and widening a neural network.

BID27 built upon this work, dubbing function-preserving operators on neural networks network morphisms.

Formally, a network morphism is a network operator satisfying N w (x) = (T N )w(x) for every x ∈ X , i.e., N w and (T N )w represent the same function.

This can be achieved by properly initializingw.

We now describe the operators used in LEMONADE and how they can be formulated as a network morphism.

We refer to Appendix A.1.1 for details.1.

Inserting a Conv-BatchNorm-ReLU block.

We initialize the convolution to be an identity mapping, as done by Chen et al. FORMULA6 ("Net2DeeperNet").

Offset and scale of BatchNormalization are initialized to be the (moving) batch mean and (moving) batch variance, hence initially again an identity mapping.

Since the ReLU activation is idempotent, i.e., ReLU (ReLU (x)) = ReLU (x), we can add it on top of the previous two operations without any further changes, assuming that the block will be added on top of a ReLU layer.2.

Increase the number of filters of a convolution.

This operator requires the layer to be changed to have a subsequent convolutional layer, whose parameters are padded with 0's.

Alternatively, one could use the "Net2WiderNet" operator by Chen et al. (2015) .3.

Add a skip connection.

We allow skip connection either by concatenation BID7 or by addition (He et al., 2016) .

In the former case, we again use zero-padding in sub-sequential convolutional layers.

In the latter case, we do not simply add two outputs x and y but rather use a convex combination (1 − λ)x + λy, with a learnable parameter λ initialized as 0 (assuming x is the original output and y the output of an earlier layer).

One common property of all network morphisms is that they can only increase the capacity of a network 1 .

This may be a reasonable property if one solely aims at finding a neural architectures with maximal accuracy, but not if one also aims at neural architectures with low resource requirements.

Also, decisions once made can not be reverted.

Operators like removing a layer could considerably decrease the resources required by the model while (potentially) preserving its performance.

Hence, we now generalize the concept of network morphisms to also cover operators that reduce the capacity of a neural architecture.

We say an operator T is an approximate network morphism (ANM) with respect to a neural network N w with parameters w if N w (x) ≈ (T N )w(x) for every x ∈ X .

We refer to Appendix A.1.2 for a formal definition.

In practice we simply determinew so thatÑ approximates N by using knowledge distillation BID3 .In our experiments, we employ the following ANM's: (i) remove a randomly chosen layer or a skip connection, (ii) prune a randomly chosen convolutional layer (i.e., remove 1/2 or 1/4 of its filters), and (iii) substitute a randomly chosen convolution by a depthwise separable convolution.

Note that these operators could easily be extended by sophisticated methods for compressing neural networks BID8 BID31 .

LEMONADE maintains a population of trained networks that constitute a Pareto front in the multi-objective space.

Parents are selected from the population inversely proportional to their density.

Children are generated by mutation operators with Lamarckian inheritance that are realized by network morphisms and approximate network morphisms.

NM operators generate children with the same initial error as their parent.

In contrast, children generated with ANM operators may incur a (small) increase in error compared to their parent.

However, their initial error is typically still very small. (Right) Only a subset of the generated children is accepted for training.

After training, the performance of the children is evaluated and the population is updated to be the Pareto front.

Algorithm 1 LEMONADE 1: input: P 0 , f, n gen , n pc , n ac 2: P ← P 0 3: for i ← 1, . . .

, n gen do 4: DISPLAYFORM0 Compute parent distribution p P (Eq. 1) BID38 : DISPLAYFORM1 Compute children distribution p child (Eq. 2) 8: DISPLAYFORM2 Evaluate f exp for N c ∈ N c ac 10:P ← P aretoF ront(P ∪ N c ac , f) 11: end for 12: return P In this section, we propose a Lamarckian Evolutionary algorithm for MultiObjective Neural Architecture DEsign, dubbed LEMONADE.

We refer to FIG1 for an illustration as well as Algorithm 1 for pseudo code.

LEMONADE aims at minimizing multiple objectives DISPLAYFORM3 denote expensive-to-evaluate objectives (such as the validation error or some measure only be obtainable by expensive simulation) and its other components f cheap (N ) ∈ R n denote cheap-to-evaluate objectives (such as model size) that one also tries to minimize.

LEMONADE maintains a population P of parent networks, which we choose to comprise all non-dominated networks with respect to f, i.e., the current approximation of the Pareto front 2 .

In every iteration of LEMONADE, we first sample parent networks with respect to some probability distribution based on the cheap objectives and generate child networks by applying network operators (described in Section 3).

In a second sampling stage, we sample a subset of children, again based on cheap objectives, and solely this subset is evaluated on the expensive objectives.

Hence, we exploit that f cheap is cheap to evaluate in order to bias both sampling processes towards areas of f cheap that are sparsely populated.

We thereby evaluate f cheap many times in order to end up with a diverse set of children in sparsely populated regions of the objective space, but evaluate f exp only a few times.

More specifically, LEMONADE first computes a density estimator p KDE (e.g., in our case, a kernel density estimator) on the cheap objective values of the current population, {f cheap (N )|N ∈ P}. Note that we explicitly only compute the KDE with respect to f cheap rather than f as this allows to evaluate p KDE (f cheap (N )) very quickly.

Then, larger number n pc of proposed children N c pc = {N c 1 , . . .

, N c npc } is generated by applying network operators, where the parent N for each child is sampled according to a distribution inversely proportional to p KDE , DISPLAYFORM4 with a normalization constant c = N ∈P 1/p KDE (f cheap (N )) −1 .

Since children have similar objective values as their parents (network morphisms do not change architectures drastically), this sampling distribution of the parents is more likely to also generate children in less dense regions of f cheap .

Afterwards, we again employ p KDE to sample a subset of n ac accepted children N c ac ⊂ N c pc .

The probability of a child being accepted is DISPLAYFORM5 withĉ being another normalization constant.

Only these accepted children are evaluated according to f exp .

By this two-staged sampling strategy we generate and evaluate more children that have the potential to fill gaps in f. We refer to the ablation study in Appendix A.2.2 for an empirical comparison of this sampling strategy to uniform sampling.

Finally, LEMONADE computes the Pareto front from the current generation and the generated children, yielding the next generation.

The described procedure is repeated for a prespecified number of generations (100 in our experiments).

We present results for LEMONADE on searching neural architectures for CIFAR-10.

We ran LEMONADE with three different settings: (i) we optimize 5 objectives and search for entire architectures (Section 5.1), (ii) we optimize 2 objectives and search for entire architectures (Appendix A.2), and (iii) we optimize 2 objectives and search for cells (Section 5.2, Appendix A.2).

We also transfer the discovered cells from the last setting to ImageNet (Section 5.4) and its down-scaled version ImageNet64x64 (Chrabaszcz et al., 2017) (Section 5.3).

All experimental details, such as a description of the search spaces and hyperparameters can be found in Appendix A.3.The progress of LEMONADE for setting (ii) is visualized in FIG2 .

The Pareto front improves over time, reducing the validation error while covering a wide regime of, e.g., model parameters, ranging from 10 000 to 10 000 000.

We aim at solving the following multi-objective problem: minimize the five objectives (i) performance on CIFAR-10 (expensive objective), (ii) performance on CIFAR-100 (expensive), (iii) number of parameters (cheap), (iv) number of multiply-add operations (cheap) and (v) inference time 3 (cheap).

We think having five objectives is a realistic scenario for most NAS applications.

Note that one could easily use other, more sophisticated measures for resource efficiency.

In this experiment, we search for entire neural network architectures (denoted as Search Space I, see Appendix A.3.2 for details) instead of convolutional cells (which we will do in a later experiment).

LEMONADE natively handles this unconstrained, arbitrarily large search space, whereas other methods are by design a-priori restricted to relatively small search spaces (Bender et al., 2018; BID15 .

Also, LEMONADE is initialized with trivial architectures (see Appendix A.3.2) rather than networks that already yield state-of-the-art performance (Cai et al., 2018b; Dong et al., 2018) .

The set of operators to generate child networks we consider in our experiments are the three network morphism operators (insert convolution, insert skip connection, increase number of filters), as well as the three approximate network morphism operators (remove layer, prune filters, replace layer) described in Section 3.

The operators are sampled uniformly at random to generate children.

The experiment ran for approximately 5 days on 16 GPUs in parallel.

The resulting Pareto front consists of approximately 300 neural network architectures.

We compare against different-sized NASNets and MobileNets V2 BID22 .

In order to ensure that differences in test error are actually caused by differences in the discovered architectures rather than different training conditions, we retrained all architectures from scratch using exactly the same optimization pipeline with the same hyperparameters.

We do not use stochastic regularization techniques, such as Shake-Shake (Gastaldi, 2017) or ScheduledDropPath in this experiment as they are not applicable to all networks out of the box.

The results are visualized in FIG3 .

As one would expect, the performance on CIFAR-10 and CIFAR-100 is highly correlated, hence the resulting Pareto fronts only consist of a few elements and differences are rather small (top left).

When considering the performance on CIFAR-10 versus the number of parameters (top right) or multiply-add operations (bottom left), LEMONADE is on par with NASNets and MobileNets V2 for resource-intensive models while it outperforms them in the area of very efficient models (e.g., less than 100,000 parameters).

In terms of inference time (bottom right), LEMONADE clearly finds models superior to the baselines.

We highlight that this result has been achieved based on using only 80 GPU days for LEMONADE compared to 2000 in and with a significantly more complex Search Space I (since the entire architecture was optimized and not only a convolutional cell).We refer to Appendix A.2 for an experiment with additional baselines (e.g., random search) and an ablation study.

Above, we compared different models when trained with the exact same data augmentation and training pipeline.

We now also briefly compare LEMONADE's performance to results reported in the literature.

We apply two widely used methods to improve results over the training pipeline used above: (i) instead of searching for entire architectures, we search for cells that are employed within a hand-crafted macro architecture, meaning one replaces repeating building blocks in the architecture with discovered cells (Cai et al., 2018b; Dong et al., 2018) and (ii) using stochastic regularization techniques, such as ScheduledDropPath during training BID19 Cai et al., 2018b) .

In our case, we run LEMONADE to search for cells within the ShakeShake macro architecture (i.e., we replace basic convolutional blocks with cells) and also use ShakeShake regularization (Gastaldi, 2017) .

Table 1 .

LEMONADE is on par or outperforms DPP-Net across all parameter regimes.

As all other methods solely optimize for accuracy, they do not evaluate models with few parameters.

However, also for larger models, LEMONADE is competitive to methods that require significantly more computational resources or start their search with non-trivial architectures (Cai et al., 2018b; Dong et al., 2018 ).

To study the transferability of the discovered cells to a different dataset (without having to run architecture search itself on the target dataset), we built architectures suited for ImageNet64x64 (Chrabaszcz et al., 2017) based on five cells discovered on CIFAR-10.

We vary (1) the number of cells per block and (2) the number of filters in the last block to obtain different architectures for a single cell (as done by for NASNets).

We compare against different sized MobileNets V2, NASNets and Wide Residual Networks (WRNs) BID29 .

For direct comparability, we again train all architectures in the same way.

In FIG4 , we plot the Pareto Front from all cells combined, as well as the Pareto Front from a single cell, Cell 2, against the baselines.

Both clearly dominate NASNets, WRNs and MobileNets V2 over the entire parameter range, showing that a multi-objective search again is beneficial.

We also evaluated one discovered cell, Cell 2, on the regular ImageNet benchmark for the "mobile setting" (i.e., networks with 4M to 6M parameters and less than 600M multiply-add operations).

The cell found by LEMONADE achieved a top-1 error of 28.3% and a top-5 error of 9.6%; this is slightly worse than published results for, e.g., NASNet (26% and 8.4%, respectively) but still competitive, especially seeing that (due to time and resource constraints), we used an off-the-shelf training pipeline, on a single GPU (for four weeks), and did not alter any hyperparameters.

We believe that our cell could perform substantially better with a better optimization pipeline and properly tuned hyperparameters (as in many other NAS papers by authors with more compute resources).

We have proposed LEMONADE, a multi-objective evolutionary algorithm for architecture search.

The algorithm employs a Lamarckian inheritance mechanism based on (approximate) network morphism operators to speed up the training of novel architectures.

Moreover, LEMONADE exploits the fact that evaluating several objectives, such as the performance of a neural network, is orders of magnitude more expensive than evaluating, e.g., a model's number of parameters.

Experiments on CIFAR-10 and ImageNet64x64 show that LEMONADE is able to find competitive models and cells both in terms of accuracy and of resource efficiency.

We believe that using more sophisticated concepts from the multi-objective evolutionary algorithms literature and using other network operators (e.g., crossovers and advanced compression methods) could further improve LEMONADE's performance in the future.

In the following two subsections we give some detailed information on the network morphisms and approximate network morphisms employed in our work.

A network morphism is a network operator satisfying the network morphism equation: DISPLAYFORM0 withw i = (w i , A, b).

The network morphism equation FORMULA8 then holds for A = 1, b = 0.

This morphism can be used to add a fully-connected or convolutional layer, as these layers are simply linear mappings.

Chen et al. (2015) dubbed this morphism "Net2DeeperNet".

Alternatively to the above replacement, one could also choosẽ DISPLAYFORM1 DISPLAYFORM2 A, b are fixed, non-learnable.

In this case, network morphism Equation ( DISPLAYFORM3 with an arbitrary functionh wh (x).

The new parameters arew i = (w i , wh,Ã).

Again, Equation (3) can trivially be satisfied by settingÃ = 0.

We can think of two modifications of a neural network that can be expressed by this morphism: firstly, a layer can be widened (i.e., increasing the number of units in a fully connected layer or the number of channels in a CNN -the Net2WiderNet transformation of Chen et al. FORMULA6 ).

Let h(x) be the layer to be widened.

For example, we can then seth = h to simply double the width.

Secondly, skip-connections by concatenation as used by BID5 can also be expressed.

If h(x) itself is a sequence of layers, h(x) = h n (x) • · · · • h 0 (x), then one could chooseh(x) = x to realize a skip from h 0 to the layer subsequent to h n .Network morphism Type III.

By definition, every idempotent function N wi i can simply be replaced by DISPLAYFORM4 with the initializationw i = w i .

This trivially also holds for idempotent functions without weights, e.g., ReLU.Network morphism Type IV.

Every layer N wi i is replaceable bỹ Nw DISPLAYFORM5 with an arbitrary function h and Equation FORMULA8 holds if the learnable parameter λ is initialized as 1.

This morphism can be used to incorporate any function, especially any non-linearity.

For example, BID27 use a special case of this operator to deal with non-linear, non-idempotent activation functions.

Another example would be the insertion of an additive skip connection, which were proposed by He et al. (2016) to the layer subsequent to N wi n in .

Note that every combination of network morphisms again yields a network morphism.

Hence, one could, for example, add a block "Conv-BatchNorm-ReLU" subsequent to a ReLU layer by using Equations FORMULA9 , FORMULA10 and (7) .

LEMONADE essentially consists of three components: (i) additionally using approximate network morphism operators to also allow shrinking architectures, (ii) using Lamarckism, i.e., (approximate) network morphisms, to avoid training from scratch, and (iii) the two-staged sampling strategy.

In Figure 6 , we present results for deactivating each of these components one at a time.

The result shows that all three components improve LEMONADE's performance.

In this section we list all the experimental details.

Search Space I corresponds to searching for an entire architecture (rather than cells).

LEMONADE's Pareto front was initialized to contain four simple convolutional networks with relatively large validation errors of 30 − 50%.

All four initial networks had the following structure: three Conv- Figure 6 : Ablation study on CIFAR-10.

We deactivate different components of LEMONADE and investigate the impact.

LEMONADE default: Performance of LEMONADE as proposed in this work.

LEMONADE no ANM: we deactivated the approximate network morphisms operators, i.e., networks can only grow in size.

LEMONADE no Lamarckism: all networks are initialized from scratch instead by means of (approximate) network morphisms.

LEMONADE no KDE: we deactivate the proposed sampling strategy and use uniform sampling of parents and children instead.

BatchNorm-ReLU blocks with intermittent Max-Pooling, followed by a global average pooling and a fully-connected layer with softmax activation.

The networks differ in the number of channels in the convolutions, and for further diversity two of them used depthwise-separable convolutions.

The models had 15 000, 50 000, 100 000 and 400 000 parameters, respectively.

For generating children in LEMONADE, we chose the number of operators that are applied to parents uniformly from {1,2,3}.LEMONADE natively handles this unconstrained, arbitrary large search space, whereas other methods are by design restricted a-priori to relatively small search spaces (Bender et al., 2018; BID15 .We restricted the space of neural architectures such that every architecture must contain at least 3 (depthwise separable) convolutions with a minimum number of filters, which lead to a lower bound on the number of parameters of approximately 10 000.The network operators implicitly define the search space, we do not limit the size of discovered architectures.

Search Space II consists of convolutional cells that are used within some macro architecture to build the neural network.

In the experiments in Section 5, we use cells within the macro architecture of the Shake-Shake architecture (Gastaldi, 2017) , whereas in the baseline experiment in the appendix (Section A.2), we rely on a simpler scheme as in as in BID13 , i.e., sequentially stacking cells.

We only choose a single operator to generate children, but the operator is applied to all occurrences of the cell in the architecture.

The Pareto Front was again initialized with four trivial cells: the first two cells consist of a single convolutional layer (followed by BatchNorm and ReLU) with F = 128 and F = 256 filters in the last block, respectively.

The other two cells consist of a single depthwise separable convolution (followed by BatchNorm and ReLU), again with either F = 128 or F = 256 filters.

To classify CIFAR-10 with MobileNets V1 and V2, we replaced three blocks with stride 2 with identical blocks with stride 1 to adapt the networks to the lower spatial resolution of the input.

We chose the replaced blocks so that there are the same number of stride 1 blocks between all stride 2 blocks.

We varied the size of MobileNets V1 and V2 by varying the width multiplier α ∈ {0.1, 0.2, . . .

, 1.2} and NASNets by varying the number of cell per block (∈ {2, 4, 6, 8}) and number of filters (∈ {96, 192, 384, 768, 1536}) in the last block.

We apply the standard data augmentation scheme described by BID16 , as well as the recently proposed methods mixup BID30 and Cutout (Devries & Taylor, 2017) .

The training set is split up in a training (45.000) and a validation (5.000) set for the purpose of architecture search.

We use weight decay (5 · 10 −4 ) for all models.

We use batch size 64 throughout all experiments.

During architecture search as well as for generating the random search baseline, all models are trained for 20 epochs using SGD with cosine annealing BID16 , decaying the learning rate from 0.01 to 0.

For evaluating the test performance, all models are trained from scratch on the training and validation set with the same setup as described above except for 1) we train for 600 epochs and 2) the initial learning rate is set to 0.025.

While searching for convolutional cells on CIFAR-10, LEMONADE ran for approximately 56 GPU days.

However, there were no significant changes in the Pareto front after approximately 24 GPU days.

The training setup (both during architecture search and final evaluation) is exactly the same as before.

The training setup on ImageNet64x64 is identical to Chrabaszcz et al. (2017) .

Below we list some additional figures.

@highlight

We propose a method for efficient Multi-Objective Neural Architecture Search based on Lamarckian inheritance and evolutionary algorithms.