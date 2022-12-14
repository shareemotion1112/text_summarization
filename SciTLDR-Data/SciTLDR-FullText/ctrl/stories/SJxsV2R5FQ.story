We present a representation for describing transition models in complex uncertain domains using relational rules.

For any action, a rule selects a set of relevant objects and computes a distribution over properties of just those objects in the resulting state given their properties in the previous state.

An iterative greedy algorithm is used to construct a set of deictic references that determine which objects are relevant in any given state.

Feed-forward neural networks are used to learn the transition distribution on the relevant objects' properties.

This strategy is demonstrated to be both more versatile and more sample efficient than learning a monolithic transition model in a simulated domain in which a robot pushes stacks of objects on a cluttered table.

Many complex domains are appropriately described in terms of sets of objects, properties of those objects, and relations among them.

We are interested in the problem of taking actions to change the state of such complex systems, in order to achieve some objective.

To do this, we require a transition model, which describes the system state that results from taking a particular action, given the previous system state.

In many important domains, ranging from interacting with physical objects to managing the operations of an airline, actions have localized effects: they may change the state of the object(s) being directly operated on, as well as some objects that are related to those objects in important ways, but will generally not affect the vast majority of other objects.

In this paper, we present a strategy for learning state-transition models that embodies these assumptions.

We structure our model in terms of rules, each of which only depends on and affects the properties and relations among a small number of objects in the domain, and only very few of which may apply for characterizing the effects of any given action.

Our primary focus is on learning the kernel of a rule: that is, the set of objects that it depends on and affects.

At a moderate level of abstraction, most actions taken by an intentional system are inherently directly parametrized by at least one object that is being operated on: a robot pushes a block, an airport management system reschedules a flight, an automated assistant commits to a venue for a meeting.

It is clear that properties of these "direct" objects are likely to be relevant to predicting the action's effects and that some properties of these objects will be changed.

But how can we characterize which other objects, out of all the objects in a household or airline network, are relevant for prediction or likely to be affected?To do so, we make use of the notion of a deictic reference.

In linguistics, a deictic (literally meaning "pointing") reference, is a way of naming an object in terms of its relationship to the current situation rather than in global terms.

So, "the object I am pushing," "all the objects on the table nearest me," and "the object on top of the object I am pushing" are all deictic references.

This style of reference was introduced as a representation strategy for AI systems by BID0 , under the name indexical-functional representations, for the purpose of compactly describing policies for a video-game agent, and has been in occasional use since then.

We will learn a set of deictic references, for each rule, that characterize, relative to the object(s) being operated on, which other objects are relevant.

Given this set of relevant objects, the problem of describing the transition model on a large, variable-size domain, reduces to describing a transition model on fixed-length vectors characterizing the relevant objects and their properties and relations, which we represent and learn using standard feed-forward neural networks.

Next, we briefly survey related work, describe the problem more formally, and then provide an algorithm for learning both the structure, in terms of deictic references, and parameters, in terms of neural networks, of a sparse relational transition model.

We go on to demonstrate this algorithm in a simulated robot-manipulation domain in which the robot pushes objects on a cluttered table.

Rule learning has a long history in artificial intelligence.

The novelty in our approach is the combination of learning discrete structures with flexible parametrized models in the form of neural networks.

Rule learning We are inspired by very early work on rule learning by BID4 , which sought to find predictive rules in simple noisy domains, using Boolean combinations of binary input features to predict the effects of actions.

This approach has a modern re-interpretation in the form of schema networks BID5 .

The rules we learn are lifted, in the sense that they can be applied to objects, generally, and are not tied to specific bits or objects in the input representation and probabilistic, in the sense that they make a distributional prediction about the outcome.

In these senses, this work is similar to that of Pasula et al. (2007) and methods that build on it ( BID9 , BID8 , BID6 .)

In addition, the approach of learning to use deictic expressions was inspired by Pasula et al. and used also by BID7 in the form of object-oriented reinforcement learning and by BID2 .

BID2 , however, relies on a full description of the states in ground first-order logic and does not have a mechanism to introduce new deictic references to the action model.

Our representation and learning algorithm improves on the Pasula et al. strategy by using the power of feed-forward neural networks as a local transition model, which allows us to address domains with real-valued properties and much more complex dependencies.

In addition, our EM-based learning algorithm presents a much smoother space in which to optimize, making the overall learning faster and more robust.

We do not, however, construct new functional terms during learning; that would be an avenue for future work for us.

Graph network models There has recently been a great deal of work on learning graph-structured (neural) network models BID1 .

There is a way in which our rule-based structure could be interpreted as a kind of graph network, although it is fairly non-standard.

We can understand each object as being a node in the network, and the deictic functions as being labeled directed hyper-edges (between sets of objects).

Unlike the typical graph network models, we do not condition on a fixed set of neighboring nodes and edges to compute the next value of a node; in fact, a focus of our learning method is to determine which neighbors (and neighbors of neighbors, etc.) to condition on, depending on the current state of the edge labels.

This means that the relevant neighborhood structure of any node changes dynamically over time, as the state of the system changes.

This style of graph network is not inherently better or worse than others: it makes a different set of assumptions (including a strong default that most objects do not change state on any given step and the dynamic nature of the neighborhoods) which are particularly appropriate for modeling an agent's interactions with a complex environment using actions that have relatively local effects.

We assume we are working on a class of problems in which the domain is appropriately described in terms of objects.

This method might not be appropriate for a single high-dimensional system in which the transition model is not sparse or factorable, or can be factored along different lines (such as a spatial decomposition) rather than along the lines of objects and properties.

We also assume a set of primitive actions defined in terms of control programs that can be executed to make actual changes in the world state and then return.

These might be robot motor skills (grasping or pushing an object) or virtual actions (placing an order or announcing a meeting).

In this section, we formalize this class of problems, define a new rule structure for specifying probabilistic transition models for these problems, and articulate an objective function for estimating these models from data.

A problem domain is given by tuple D = (??, P, F, A) where ?? is a countably infinite universe of possible objects, P is a finite set of properties P i : ?? ??? R, i ??? [N P ] = {1, ?? ?? ?? , N P }, and F is a finite set of deictic reference functions DISPLAYFORM0 where ??? (??) denotes the powerset of ??. Each function F i ??? ?? maps from an ordered list of objects to a set of objects, and we define it as DISPLAYFORM1 where the relation f i : ?? mi+1 ??? {True, False} is defined in terms of the object properties in P. For example, if we have a location property P loc and m i = 1, we can define f i (o, o 1 ) = 1 Ploc(o)???Ploc(o1) <0.5 so that the function F i associated with f i maps from one object to the set of objects that are within 0.5 distance of its center; here 1 is an indicator function.

Finally, A is a set of action templates DISPLAYFORM2 where ?? is the space of executable control programs.

Each action template is a function parameterized by continuous parameters ?? i ??? R di and a tuple of n i objects that the action operates on.

In this work, we assume that P, F and A are given.

A problem instance is characterized by I = (D, U), where D is a domain defined above and U ??? ?? is a finite universe of objects with |U| = N U .

For simplicity, we assume that, for a particular instance, the universe of objects remains constant over time.

In the problem instance I, we characterize a state s in terms of the concrete values of all properties in P on all objects in U; that is, DISPLAYFORM0 A problem instance induces the definition of its action space A, constructed by applying every action template A i ??? A to all tuples of n i elements in U and all assignments ?? i to the continuous parameters; namely, DISPLAYFORM1

In many domains, there is substantial uncertainty, and the key to robust behavior is the ability to model this uncertainty and make plans that respect it.

A sparse relational transition model (SPARE) for a domain D, when applied to a problem instance I for that domain, defines a probability density function on the resulting state s resulting from taking action a in state s.

Our objective is to specify this function in terms of domain elements P, R, and F in such a way that it will apply to any problem instance, independent of the number and properties of the objects in its universe.

We achieve this by defining the transition model in terms of a set of transition rules, T = {T k } K k=1 and a score function C : T ?? S ??? N. The score function takes in as input a state s and a rule T ??? T , and outputs a non-negative integer.

If the output is 0, the rule does not apply; otherwise, the rule can predict the distribution of the next state to be p(s | s, a; T ).

The final prediction of SPARE is DISPLAYFORM0 whereT = arg max T ???T C(T, s) and the matrix DISPLAYFORM1 ) is the default predicted covariance for any state that is not predicted to change, so that our problem is well-formed in the presence of noise in the input.

Here I N U is an identity matrix of size N U , and diag ([?? DISPLAYFORM2 represents a square diagonal matrix with ?? i on the main diagonal, denoting the default variance for property P i if no rule applies.

Note that the transition rules will be learned from past experience with a loss function specified in Section 3.3.

In the rest of this section, we formalize the definition of transition rules and the score function.

Transition rule T = (A, ??, ???, ?? ?? , v default ) is characterized by an action template A, two ordered lists of deictic references ?? and ??? of size N ?? and N ??? , a predictor ?? ?? and the default variances DISPLAYFORM3 for each property P i under this rule.

The action template is defined as operating on a tuple of n object variables, which we will refer to as DISPLAYFORM4 A reference list uses functions to designate a list of additional objects or sets of objects, by making deictic references DISPLAYFORM5 Figure 2: Instead of directly mapping from current state s to next state s , our prediction model uses deictic references to find subsets of objects for prediction.

In the left most graph, we illustrate what relations are used to construct the input objects with two rules for the same action template, DISPLAYFORM6 default ), where the reference list DISPLAYFORM7 1 to the target object o 2 and added input features computed by an aggregator g on o 3 , o 6 to the inputs of the predictor of rule T 1 .

Similarly for DISPLAYFORM8 , the first deictic reference selected o 3 and then ??2 is applied on o 3 to get o 1 .

The predictors ?? ?? are neural networks that map the fixed-length input to a fixed-length output, which is applied to a set of objects computed from a relational graph on all the objects, derived from the reference list DISPLAYFORM9 , to compute the whole next state s .

Because ?? DISPLAYFORM10 ?? is only predicting a single property, we use a "de-aggregator" function h to assign its prediction to both objects o 4 , o 6 .

based on previously designated objects.

In particular, ?? generates a list of objects whose properties affect the prediction made by the transition rule, while ??? generates a list of objects whose properties are affected after taking an action specified by the action template A.We begin with the simple case in which every function returns a single object, then extend our definition to the case of sets.

Concretely, for the t-th element DISPLAYFORM11 where F ??? F is a deictic reference function in the domain, m is the arity of that function, and integer k j ??? [n+t???1] specifies that object O n+t in the object list can be determined by applying function F to objects (O kj ) m j=1 .

Thus, we get a new list of objects, DISPLAYFORM12 .

So, reference ?? 1 can only refer to the objects (O i ) n i=1 that are named in the action, and determines an object O n+1 .

Then, reference ?? 2 can refer to objects named in the action or those that were determined by reference ?? 1 , and so on.

When the function DISPLAYFORM13 ) ??? ?? returns a set of objects rather than a single object, this process of adding more objects remains almost the same, except that the O t may denote sets of objects, and the functions that are applied to them must be able to operate on sets.

In the case that a function F returns a set, it must also specify an aggregator, g, that can return a single value for each property P i ??? P, aggregated over the set.

Examples of aggregators include the mean or maximum values or possibly simply the cardinality of the set.

For example, consider the case of pushing the bottom (block A) of a stack of 4 blocks, depicted in FIG1 .

Suppose the deictic reference is F =above, which takes one object and returns a set of objects immediately on top of the input object.

Then, by applying F =above starting from the initial set O 0 = {A}, we get an ordered list of sets of objects DISPLAYFORM14 Returning to the definition of a transition rule, we now can see informally that if the parameters of action template A are instantiated to actual objects in a problem instance, then ?? and ??? can be used to determine lists of input and output objects (or sets of objects).

We can use these lists, finally, to construct input and output vectors.

The input vector x consists of the continuous action parameters ?? of action A and property P i (O t ) for all properties P i ??? P and objects O t ??? O N?? that are selected by ?? in arbitrary but fixed order.

In the case that O t is a set of size greater than one, the aggregator associated with the function F that computed the reference is used to compute P i (O t ).

Similar for the desired output construction, we use the references in the list ???, initialize?? (0) = O (0) , and gradually add more objects to construct the output set of objects?? =?? (N???) .

The output vector is y = [P (??)]?? ?????,P ???P where if?? is a set of objects, we apply a mean aggregator on the properties of all the objects in??.

The predictor ?? ?? is some functional form ?? (such as a feed-forward neural network) with parameters (weights) ?? that will take values x as input and predict a distribution for the output vector y.

It is difficult to represent arbitrarily complex distributions over output values.

In this work, we restrict ourselves to representing a Gaussian distributions on all property values in y, encoded with a mean and independent variance for each dimension.

Now, we describe how a transition rule can be used to map a state and action into a distribution over the new state.

A transition rule T = (A, ??, ???, ?? ?? , v default ) applies to a particular state-action (s, a) pair if a is an instance of A and if none of the elements of the input or output object lists is empty.

To construct the input (and output) list, we begin by assigning the actual objects o 1 , . . .

, o n to the object variables O 1 , . . .

, O n in action instance a, and then successively computing references ?? i ??? ?? based on the previously selected objects, applying the definition of the deictic reference F in each ?? i to the actual values of the properties as specified in the state s.

If, at any point, a ?? i ??? ?? or ?? i ??? ??? returns an empty set, then the transition rule does not apply.

If the rule does apply, and successfully selects input and output object lists, then the values of the input vector x can be extracted from s, and predictions are made on the mean and variance values Pr( DISPLAYFORM15 ??2 (x) be the vector entry corresponding to the predicted Gaussian parameters of property P i of j-th output object set?? j and denote s[o, P i ] as the property P i of object o in state s, for all o ??? U. The predicted distribution of the resulting state p(s | s, a; T ) is computed as follows: DISPLAYFORM16 where v i ??? v default is the default variance of property P i in rule T .

There are two important points to note.

First, it is possible for the same object to appear in the object-list more than once, and therefore for more than one predicted distribution to appear for its properties in the output vector.

In this case, we use the mixture of all the predicted distributions with uniform weights.

Second, when an element of the output object list is a set, then we treat this as predicting the same single property distribution for all elements of that set.

This strategy has sufficed for our current set of examples, but an alternative choice would be to make the predicted values be changes to the current property value, rather than new absolute values.

Then, for example, moving all of the objects on top of a tray could easily specify a change to each of their poses.

We illustrate how we can use transition rules to build a SPARE in Fig. 2 .For each transition rule T k ??? T and state s ??? S, we assign the score function value to be 0 if T k does not apply to state s. Otherwise, we assign the total number of deictic references plus one, N ?? + N ??? + 1, as the score.

The more references there are in a rule that is applicable to the state, the more detailed the match is between the rules conditions and the state, and the more specific the predictions we expect it to be able to make.

We frame the problem of learning a transition model from data in terms of conditional likelihood.

The learning problem is, given a problem domain description D and a set of experience E tuples, DISPLAYFORM0 , find a SPARE T that minimizes the loss function: DISPLAYFORM1 Note that we require all of the tuples in E to belong to the same domain D, and require for any ( DISPLAYFORM2 and s (i) belong to the same problem instance, but individual tuples may be drawn from different problem instances (with, for example, different numbers and types of objects).

In fact, to get good generalization performance, it will be important to vary these aspects across training instances.

We describe our learning algorithm in three parts.

First, we introduce our strategy for learning ?? ?? , which predicts a Gaussian distribution on y, given x. Then, we describe our algorithm for learning reference lists ?? and ??? for a single transition rule, which enable the extraction of x and y from E. Finally, we present an EM method for learning multiple rules.

For a particular transition rule T with associated action template A, once ?? and ??? have been specified, we can extract input and output features x and y from a given set of experience samples E. We would like to learn the transition rule's predictor ?? ?? to minimize Eq. (2).

Our predictor takes the form ?? ?? (x) = N (?? ?? (x), ?? ?? (x)) and a neural network is used to predict both the mean ?? ?? (x) and the diagonal variance ?? ?? (x).

We directly optimize the negative data-likelihood loss function DISPLAYFORM0 Let E T ??? E be the set of experience tuples to which rule T applies.

Then once we have ??, we can optimize the default variance of the rule DISPLAYFORM1 It can be shown that these loss-minimizing values for the default predicted variances v default are the empirical averages of the squared deviations for all unpredicted objects (i.e., those for which ?? ?? does not explicitly make predictions), where averages are computed separately for each object property.

We use ??, v default ??? LEARNDIST(D, E, ??, ???) to refer to this learning and optimization procedure for the predictor parameters and default variance.

Algorithm 1 Greedy procedure for constructing ??. DISPLAYFORM0 train model using ?? 0 = ???, save loss L 0

i ??? 1 4: DISPLAYFORM0 for all ?? ??? R i do 7: DISPLAYFORM1 else breakIn the simple setting where only one transition rule T exists in our domain D, we show how to construct the input and output reference lists ?? and ??? that will determine the vectors x and y. Suppose for now that ??? and v default are fixed, and we wish to learn ??. Our approach is to incrementally build up ?? by adding DISPLAYFORM2 ) tuples one at a time via a greedy selection procedure.

Specifically, let R i be the universe of possible ?? i , split the experience samples E into a training set E train and a validation set E val , and initialize the list ?? to be ?? 0 = ???. For each i, compute ?? i = arg min ?????Ri L(T ?? ; D, E val ), where L in Eq. (2) evaluates a SPARE T ?? with a single transition rule T = (A, ?? i???1 ??? {??}, ???, ?? ?? , v default ), where ?? and v default are computed using the LEARNDIST described in Section 4.1 2 .

If the value of the loss function L(T ??i ; D, E val ) is less than the value of L(T ??i???1 ; D, E val ), then we let ?? i = ?? i???1 ???{?? i } and continue.

Else, we terminate the greedy selection process with ?? = ?? i???1 , since further growing the list of deictic references hurts the loss.

We also terminate the process when i exceeds some predetermined maximum allowed number of input deictic references, N ?? .

Pseudocode for this algorithm is provided in Algorithm 1.In our experiments we set ??? = ?? and construct the lists of deictic references using a single pass of the greedy algorithm described above.

This simplification is reasonable, as the set of objects that are relevant to predicting the transition outcome often overlap substantially with the objects that are affected by the action.

Alternatively, we could learn ??? via an analogous greedy procedure nested around or, as a more efficient approach, interleaved with, the one for learning ??.

Our training data in robotic manipulation tasks are likely to be best described by many rules instead of a single one, since different combinations of relations among objects could be present in different states.

For example, we may have one rule for pushing a single object and another rule for pushing a stack of objects.

We now address the case where we wish to learn K rules from a single experience set E, for K > 1.

We do so via initial clustering to separate experience samples into K clusters, one for each rule to be learned, followed by an EM-like approach to further separate samples and simultaneously learn rule parameters.

To facilitate the learning of our model, we will additionally learn membership probabilities Z = ((z i,j ) DISPLAYFORM0 , where z i,j represents the probability that the i-th experience sample is assigned to transition rule T j , and DISPLAYFORM1 .

We initialize membership probabilities via clustering, then refine them through EM.Because the experience samples E may come from different problem instances and involve different numbers of objects, we cannot directly run a clustering algorithm such as k-means on the (s, a, s ) samples themselves.

Instead we first learn a single transition rule T = (A, ??, ???, ?? ?? , v default ) from E using the algorithm in Section 4.2, use the resulting ?? and ??? to transform E into x and y, and then run k-means clustering on the concatenation of x, y, and values of the loss function when T is used to predict each of the samples.

For each experience sample, the squared distance from the sample to each of the K cluster centers is computed, and membership probabilities for the sample to each of the K transition rules to be learned are initialized to be proportional to the (multiplicative) inverses of these squared distances.

Before introducing the EM-like algorithm that simultaneously improves the assignment of experience samples to transition rules and learns details of the rules themselves, we make a minor modification to transition rules to obtain mixture rules.

Whereas a probabilistic transition rule has been defined as T = (A, ??, ???, ?? ?? , v default ), a mixture rule is T = (A, ?? ?? , ?? ??? , ??), where ?? ?? represents a distribution over all possible lists of input references ?? (and similarly for ?? ??? and ???), of which there are a finite number, since the set of available reference functions F is finite, and there is an upper bound N ?? on the maximum number of references ?? may contain.

For simplicity of terminology, we refer to each possible list of references ?? as a shell, so ?? ?? is a distribution over possible shells.

DISPLAYFORM2 is a collection of ?? transition rules (i.e., predictors ?? ?? (k) , each with an associated ?? (k) , ??? (k) , and v (k) default ).

To make predictions for a sample (s, a) using a mixture rule, predictions from each of the mixture rule's ?? transition rules are combined according to the probabilities that ?? ?? and ?? ??? assign to each transition rule's ?? (k) and ??? (k) .

Rather than having our EM approach learn K transition rules, we instead learn K mixture rules, as the distributions ?? ?? and ?? ??? allow for smoother sorting of experience samples into clusters corresponding to the different rules, in contrast to the discrete ?? and ??? of regular transition rules.

As before, we focus on the case where for each mixture rule, DISPLAYFORM3 , and ?? ?? = ?? ??? as well.

Our EM-like algorithm is then as follows:1.

For each j ??? [K], initialize distributions ?? ?? = ?? ??? for mixture rule T j as follows.

First, use the algorithm in Section 4.2 to learn a transition rule on the weighted experience samples E Zj with weights equal to the membership probabilities DISPLAYFORM4 .

In the process of greedily assembling reference lists ?? = ???, data likelihood loss function values are computed for multiple explored shells, in addition to the shell ?? = ??? that was ultimately selected.

Initialize ?? ?? = ?? ??? to distribute weight proportionally, ac- DISPLAYFORM5 , with the summation taken over all explored shells ??, is a normalization factor so that the total weight assigned by ?? ?? to explored shells is 1??? .

The remaining probability weight is distributed uniformly across unexplored shells.

, where we have dropped subscripting according to j for notational simplicity: DISPLAYFORM0 default ) using the procedure in Section 4.2 on the weighted experience samples E Zj , where we choose DISPLAYFORM1 to be the list of references with k-th highest weight according to ?? ?? = ?? ??? .

(b) Update ?? ?? = ?? ??? by redistributing weight among the top ?? shells according to a voting procedure where each training sample "votes" for the shell whose predictor minimizes the validation loss for that sample.

In other words, the i-th experience sample DISPLAYFORM2 ).

Then, shell weights are assigned to be proportional to the sum of the sample weights (i.e., membership probability of belonging to this rule) of samples that voted for each particular shell: the number of votes received by the k-th shell is V (k) = |E| i=1 1 v(i)=k ?? z i,j , for indicator function 1 and k ??? [??].

Then, ?? ?? (k), the current k-th highest value of ?? ?? , is updated to become V (k)/??, where ?? is a normalization factor to ensure that ?? ?? remains a valid probability distribution. (Specifically, ?? = ( DISPLAYFORM3 Step 2a, in case the ?? shells with highest ?? ?? values have changed, in preparation for using the mixture rule to make predictions in the next step.3.

Update membership probabilities by scaling by data likelihoods from using each of the K rules to make predictions: DISPLAYFORM4 is the data likelihood from using mixture rule T j to make predictions for the i-th experience sample E (i) , and DISPLAYFORM5 ) is a normalization factor to maintain K j=1 z i,j = 1.

4.

Repeat Steps 2 and 3 some fixed number of times, or until convergence.

We apply our approach, SPARE, to a challenging problem of predicting pushing stacks of blocks on a cluttered table top.

We describe our domain, the baseline that we compare to and report our results.

In our domain D = (??, P, F, A), the object universe ?? is composed of blocks of different sizes and weight, the property set P includes shapes of the blocks (width, length, height) and the position of the block ((x, y, z) location relative to the table).

We have one action template, push(??, o), which pushes toward a target object o with parameters ?? = (x g , y g , z g , d), where (x g , y g , z g ) is the 3D position of the gripper before the push starts and d is the distance of the push.

The orientation of the gripper and the direction of the push are computed from the gripper location and the target object location.

We simulate this 3D domain using the physically realistic PyBullet (Coumans & Bai, 2016 simulator.

In real-world scenarios, an action cannot be executed with the exact action parameters due to the inaccuracy in the motor and hence in our simulation, we add Gaussian noise on the action parameters during execution to imitate this effect.

We consider the following deictic references in the reference collection F: (1) identity (O), which takes in an object O and returns O; (2) above (O), which takes in an object O and returns the object immediately above O; (3) below (O), which takes in an object O and returns the object immediately below O; (4) nearest (O), which takes in an object O and returns the object that is closest to O.

Neural network (NN) We compare to a neural network function approximator that takes in as input the current state s ??? R N P ??N U and action parameter ?? ??? R N A , and outputs the next state s ??? R N P ??N U .

The list of objects that appear in each state is ordered: the target objects appear first and the remaining objects are sorted by their poses (first sort by x coordinate, then y, then z).Graph NN We compare to a fully connected graph NN.

Each node of the graph corresponds to an object in the scene, and the action ?? is concatenated to the state of each object.

Bidirectional edges connect every node in the graph.

The graph NN consists of encoders for the nodes and edges, propagation networks for message passing, and a node decoder to convert back to predict the mean and variance of the next state of each object.

As a sanity check, we start from a simple problem where a gripper is pushing a stack of three blocks with two extra blocks on the table.

We randomly sampled 1250 problem instances by drawing random block shapes and locations from a uniform distribution within a range while satisfying the condition that the stack of blocks is stable and the extra blocks do not affect the push.

In each problem instance, we uniformly randomly sample the action parameters and obtain the training data, a collection of tuples of state, action and next state, where the target object of the push action is always the one at the bottom of the stack.

We held out 20% of the training data as the validation set.

We found that our approach is able to reliably select the correct combinations of the references that select all the blocks in the problem instance to construct inputs and outputs.

In FIG3 , we show how the performance varies as deictic references are added during a typical run of this experiment.

The solid purple curves show training performance, as measured by data likelihood on the validation set, while the dashed purple curve shows performance on a held-out test set with 250 unseen problem instances.

As expected, performance improves noticeably from the addition of the first two deictic references selected by the greedy selection procedure, but not from the 4th.

The brown curve shows the learned default standard deviations, used to compute data likelihoods for features of objects not explicitly predicted by the rule.

As expected, the learned default standard deviation drops as deictic references are added, until it levels off after the third reference is added since at that point the set of references captures all moving objects in the scene.

Sensitivity analysis on the number of objects We compare our approach to the baselines in terms of how sensitive the performance is to the number of objects that exist in the problem instance.

We continue the setting where a stack of three blocks lie on a table, with extra blocks that may affect the prediction of the next state.

FIG3 shows the performance, as measured by the log data likelihood, as a function of the number of extra blocks.

For each number of extra blocks, we used 1250 training problem instances with 20% as the validation set and 250 testing problem instances.

When there are no extra blocks, SPARE learns a single rule whose x and y contain the same information as the inputs and outputs for the baselines.

As more objects are added to the table, NN's performance drops as the presence of these additional objects appear to complicate the scene and NN is forced to consider more objects when making its predictions.

SPARE outperforms graph NN, as the good predictions for the extra blocks contribute to the log data likelihood.

Note that, performance aside, NN is limited to problems for which the number of objects in the scenes is fixed, as the it requires a fixed-size input vector containing information about all objects.

Our SPARE approach does not have this limitation, and could have been trained on a single, large dataset that is the combination of the datasets with varying numbers of extra objects.

However, we did not do this in our experiments for the sake of providing a more fair comparison against NN.Sample efficiency We evaluate our approach on more challenging problem instances where the robot gripper is pushing blocks on a cluttered table top and there are two additional blocks on the table that do not interfere or get affected by the pushing action.

FIG3 (c) plots the data likelihood as a function of the number of training samples.

We evaluate with training samples varying from 100 to 1250 and in each setting, the test dataset has 250 samples.

Both our approach and the baselines benefit from having more training samples, but our approach is much more sample efficient and achieves good performance within only 500 training samples.

Learning multiple transition rules Now we put our approach in a more general setting where multiple transition rules need to be learned for prediction of the next state.

Our approach adopts an EM-like procedure to assign each training sample its distribution on the transition rules and learn each transition rule with re-weighted training samples.

First, we construct a training dataset and 70% of it is on pushing 4-block stack.

Our EM approach is able to concentrate to the 4-block case as shown in FIG4 .

The three curves correspond to the three stack heights in the original dataset, and each shows the average weight assigned to the "target" rule among samples of that stack height, where the target rule is the one that starts with a high concentration of samples of that particular height.

At iteration 0, we see that the rules were initialized such that samples were assigned 70% probability of belonging to specific rules, based on stack height.

As the algorithm progresses, the samples separate further, suggesting that the algorithm is able to separate samples into the correct groups.

Conclusion These results demonstrate the power of combining relational abstraction with neural networks, to learn probabilistic state transition models for an important class of domains from very little training data.

In addition, the structural nature of the learned models will allow them to be used in factored search-based planning methods that can take advantage of sparsity of effects to plan efficiently.

Finally, the end purpose of obtaining a transition model for robotic actions is to enable planning to achieve high-level goals.

Due to time constraints, this work did not assess the effectiveness of learned template-based models in planning, but this is a promising area of future work as the assumption that features of objects for which a template makes no explicit prediction do not change meshes well with STRIPS-style planners, as they make similar assumptions.

We here provide details on our experiments and more results in experiments.

Experimental details Our experiments on SPARE in this paper used nueral network predictors for making mean predictions and variance predictions, described in Section 4.1.

Each network was implemented as a fully-connected network with two hidden layers of 64 nodes each in Keras, used ReLU activations between layers, and the Adam optimizer with default parameters.

Predictors for the templates approach were trained for 1000 epochs each with a decaying learning rate starting at 1e-2 and decreasing by a factor of 0.6 every 100 epochs.

The baseline NN predictor was implemented in exactly the same way.

For the GNN, we used a node encoder and edge encoder to map to latent spaces of 16 dimensions.

The propagation networks consisted of 2 fully connected layers of 16 units each, and the decoder mapped back to 6 dimensions: 3 for the mean, and 3 for the variance.

The GNN was trained using a decaying learning rate starting at 1e-2, and decreasing by a factor of 0.5 every 100 epochs.

A total of 900 epochs were used.

States were parameterized by the (x, y, z) pose of each object in the scene, ordered such that the target object of the action always appeared first, and other objects appeared in random order (except for the baseline).

Action parameters included the (x, y, z) starting pose of the robotic gripper, as well as a "push distance" parameter that controls how long the push action lasts.

Actions were implemented to be stochastic by adding some amount of noise to the target location of each push, potentially reflecting inaccuries in robotic control.

We use the clustering-based approaches for initializing membership probabilities presented in Section 4.3.

In this section, we how well our clustering approach performs.

TAB0 shows the sample separation achieved by the discrete clustering approach, where samples are assigned solely to their associated clusters found by k-means, on the push dataset for stacks of varying height.

Each column corresponds to the one-third of samples which involve stacks of a particular height.

Entries in the table show the proportion of samples of that stack height that have been assigned to each of the three clusters, where in generating these data the clusters were ordered so that the predominantly 2-block sample cluster came first, followed by the predominantly 3-block cluster, then the 4-block cluster.

Values in parentheses are standard deviations across three runs of the experiment.

As seen in the table, separation of samples into clusters is quite good, though 22.7% of 3-block samples were assigned to the predominantly 4-block cluster, and 11.5% of 2-block samples were assigned to the predominantly 3-block cluser.

The sample separation evidenced in TAB0 is enough such that templates trained on the three clusters of samples reliably select deictic references that consider the correct number of blocks, i.e., the 2-block cluster reliably learns a template which considers the target object and the object above the target, and similiarly for the other clusters and their respective stack heights.

However, initializing samples to belong solely to a single cluster, rather than initializing membership probabilities, is unlikely to be robust in general, so we turn to the proposed clustering-based approaches for initializing membership probabilities instead.

TAB1 is analogous to TAB0 in structure, but shows sample separation results for sample membership probabilities initialized to be proportional to the inverse distance from the sample to each of the cluster centers found by k-means.

TAB2 is the same, except with membership probabilities initialized to be proportional to the square of the inverse distance to cluster centers.

Sample separation is better in the case of squared distances than non-squared distances, but it's unclear whether this result generalizes to other datasets.

For our specific problem instance, the log data likelihood feature turns out to be very important for the success of these clustering-based initialization approaches.

For example, Table 4 shows results analogous to those in TAB2 , where the only difference is that all log data likelihoods were multiplied by five before being passed as input to the k-means clustering algorithm.

Comparing the two tables, this scaling of data likelihood to become relatively more important as a feature results in better data separation.

This suggests that the relative importance between log likelihood and the other input features is a parameter of these clustering approaches that should be tuned.

Effect of object ordering on baseline performance The single-predictor baseline used in our experiments receives all objects in the scene as input, but this leaves open the question of in what order these objects should be presented.

Because the templates approach has the target object of the action specified, in the interest of fairness this information is also provided to the baseline by having the target object always appear first in the ordering.

As there is in general no clear ordering for the remainder of the objects, we could present them in a random order, but perhaps sorting the objects according to position (first by x-coordinate, then y, then z) could result in better predictions than if objects are completely randomly ordered.

Table 4 : Sample separation from clustering-based initialization of membership probabilities, where probabiliites are assigned to be proportional to squared inverse distance to cluster centers, and log data likelihood feature used as part of k-means clustering has been multiplied by a factor of five.

Standard deviations are reported in parentheses.

To analyze the effect of object ordering on baseline performance, we run the same experiment where a push action is applied to the bottom-most of a stack of three blocks, and there exist some number of additional objects on the table that do not interfere with the action in any way.

Figure 6 shows our results.

We test three object orderings: random ("none"), sorted according to object position ("xtheny"), and an ideal ordering where the first three objects in the ordering are exactly the three objects in the stack ordered from bottom up ("stack").

As expected, in all cases, predicted log likelihood drops as more extra blocks are added to the scene, and the random ordering performs worst while the ideal ordering performs best.

Figure 6: Effect of object ordering on baseline performance, on task of pushing a stack of three blocks on a table top, where there are extra blocks on the table that do not interfere with the push.

<|TLDR|>

@highlight

A new approach that learns a representation for describing transition models in complex uncertaindomains using relational rules. 