Consider a world in which events occur that involve various entities.

Learning how to predict future events from patterns of past events becomes more difficult as we consider more types of events.

Many of the patterns detected in the dataset by an ordinary LSTM will be spurious since the number of potential pairwise correlations, for example, grows quadratically with the number of events.

We propose a type of factorial LSTM architecture where different blocks of LSTM cells are responsible for capturing different aspects of the world state.

We use Datalog rules to specify how to derive the LSTM structure from a database of facts about the entities in the world.

This is analogous to how a probabilistic relational model (Getoor & Taskar, 2007) specifies a recipe for deriving a graphical model structure from a database.

In both cases, the goal is to obtain useful inductive biases by encoding informed independence assumptions into the model.

We specifically consider the neural Hawkes process, which uses an LSTM to modulate the rate of instantaneous events in continuous time.

In both synthetic and real-world domains, we show that we obtain better generalization by using appropriate factorial designs specified by simple Datalog programs.

Temporal sequence data is abundant in applied machine learning.

A common task is to impute missing events, e.g., to predict the future from the past.

Often this is done by fitting a generative probability model.

For evenly spaced sequences, historically popular models have included hidden Markov models and discrete-time linear dynamical systems, with more recent interest in recurrent neural network models such as LSTMs.

For irregularly spaced sequences, a good starting point is the Hawkes process, a self-exciting temporal point process; many variations and enhancements have been published, including neural variants using LSTMs.

All of these models can be described schematically by Figure 1a .

Events e i , e i+1 , . . .

are assumed to be conditionally independent of previous events, given the system state s i (which may or may not be fully known given events e 1 , . . .

, e i ).

That is, s i is enough to determine the joint distribution of the i th event and the updated state s i+1 , which is needed to recursively predict all subsequent events.

Figure 1a and its caption show the three types of influence in the model.

The update, affect, and depend arrows are characterized by parameters of the model.

In the case of a recurrent neural network, these are the transition, input, and output matrices.

Our main idea in this paper is to inject structural zeros into these weight matrices.

Structural zeros are weights that are fixed at zero regardless of the model parameters.

In other words, we will remove many connections (synapses) from both the recurrent and non-recurrent portions of the neural network.

Parameter estimation must use the sparse remaining connections to explain the observed data.

Specifically, we partition the neural state s i ??? R d into a number of node blocks.

Different node blocks are intended to capture different aspects of the world's state at step i. By zeroing out rectangular blocks of the weight matrix, we will restrict how these node blocks interact with the events and with one another.

An example is depicted in Figures 1b (affect, depend) and 1d (update).

In addition, by reusing nonzero blocks within a weight matrix, we can stipulate (for example) that event e affects node block b in the same way in which event e affects node block b .

Such parameter tying makes it possible to generalize from frequent events to rare events of the same type.

Although our present experiments are small, we are motivated by the challenges of scale.

Real-world domains may have millions of event types, including many rare types.

To model organizational behavior, we might consider a dataset of meetings and emails in a large organization.

To model supply chains, we might consider purchases of goods and services around the world.

In an unrestricted model, anything in the past could potentially influence anything in the future, making estimation extremely difficult.

Structural zeroes and parameter tying, if chosen carefully, should help us avoid overfitting to coincidental patterns in the data.

Analogous architectures have been proposed in the world of graphical models and causal models.

Indeed, to write down such a model is to explicitly allow specific direct interactions and forbid the rest.

For example, the edges of a Gaussian graphical model explicitly indicate which blocks of the inverse covariance matrix are allowed to be nonzero.

Some such models reuse blocks (Hojsgaard & Lauritzen, 2008) .

As another example, a factorial HMM (Ghahramani & Jordan, 1997 )-an HMM whose states are m-tuples-can be regarded as a simple example of our architecture.

The state s i can be represented using m node blocks, each of which is a 1-hot vector that encodes the value of a different tuple element.

The key aspect of a factorial HMM is that the stochastic transition matrix (update in Figure 1d ) is fully block-diagonal.

The affect matrix is 0, since the HMM graphical model does not feed the output back into the next state; the depend matrix is unrestricted.

But how do we know which interactions to allow and which to forbid?

This is a domain-specific modeling question.

In general, we would like to exploit the observation that events are structured objects with participants (which is why the number of possible event types is often large).

For example, a travel event involves both a person and a place.

We might assume that the probability that Alice travels to Chicago depends only on Alice's state, the states of Alice's family members, and even the state of affairs in Chicago.

Given that modeling assumption, parameter estimation cannot try to derive this probability (presumably incorrectly) from the state of the coal market.

These kinds of systematic dependencies can be elegantly written down using Datalog rules, as we will show.

Datalog rules can refer to database facts, such as the fact that Alice is a person and that she is related to other people.

Given these facts, we use Datalog rules to automatically generate the set of possible events and node blocks, and the ways in which they influence one another.

Datalog makes it easy to give structured names to the events and node blocks.

The rules can inspect these structures via pattern-matching.

In short, our contribution is to show how to use a Datalog program to systematically derive a constrained neural architecture from a database.

Datalog is a blend of logic and databases, both of which have previously been used in various formalisms for deriving a graphical model architecture from a database (Getoor & Taskar, 2007) .

Our methods could be applied to RNN sequence models.

In this setting, each possible event type would derive its unnormalized probability from selected node blocks of state s i .

Normalizing these probabilities to sum to 1 would yield the model's distribution for event e i .

Only the normalizing constant would depend on all node blocks.

In this paper, we focus on the even more natural setting of real-time events.

Here no normalizing constant is needed: the events are not in competition.

As we will see in section 5.1, it is now even possible for different node blocks to generate completely independent sequences of timestamped events.

The observed dataset is formed by taking the union of these sequences.

In the real-time setting, event e i has the form k i @t i where k i ??? K is the type of the event and t i ??? R is its time.

The probability of an event of type k at any specific instant t is infinitesimal.

We will model how this infinitesimal probability depends on selected node blocks of s i .

There is no danger that two events will ever occur at the same instant, i.e., the probability of this is 0.

We begin by describing our baseline model for this setting, drawn from Mei & Eisner (2017) .

In general, a multivariate point process is a distribution over possible sequences of events e 1 = k 1 @t 1 , e 2 = k 2 @t 2 , . . .

where 0 < t 1 < t 2 < . .

..

A common paradigm for defining such processes, starting with Hawkes (1971) , is to describe their temporal evolution as in Figure 1a .

Each s i is deterministically computed from s i???1 (update) and e i???1 (affect), according to some formula, so by induction, s i is a deterministic summary of the first i ??? 1 events.

e i = k i @t i is then emitted stochastically from some distribution parameterized by s i (depend).

The structure of the depend distribution is the interesting part.

s i is used, for each event type k ??? K, to define some time-varying intensity function ?? k : (t i???1 , ???) ??? R ???0 .

This intensity function is treated as the parameter of an inhomogeneous Poisson process, which stochastically generates a set of future events of type k at various times in (t i???1 , ???).

2 Thus, all these |K| Poisson processes together give us many events of the form e = k@t.

The first such event-the one with the earliest time t-is taken to be the next event e i .

The remaining events are discarded (or in practice, never generated).

As our baseline method, we take the neural Hawkes process (Mei & Eisner, 2017) to be our method for computing s i and defining the intensity function ?? k from it.

In that work, s i actually describes a parametric function of the form h : (t i???1 , ???) ??? R d , which describes how the hidden state of the system evolves following event e i???1 .

That function is used to define the intensity functions via

2 Under an inhomogenous Poisson process, disjoint intervals generate events independently, and the number of events on the interval (a, b] is Poisson-distributed with mean b a ?? k (t) dt.

Thus, on a sufficiently narrow interval (t, t + dt], the probability of a single event is approximately ?? k (t) dt and the probability of more than one event is approximately 0, with an error of O(dt 2 ) in both cases.

so the parameters of depend are the vectors v k and the monotonic functions f k .

Once e i = k i @t i has been sampled, the parameters for s i+1 are obtained by

where ?? is inspired by the structure of an LSTM, the affect parameters are given by matrix U and the event embeddings w k , and the depend parameters are given by matrix V.

In this paper, we will show an advantage to introducing structural zeroes into v k , U, and V.

In real world, atomic events typically involve a predicate and a few arguments (called entities in the following), in which case it makes sense to decompose an event type into a structured form 3 such as email(alice,bob), travel(bob,chicago), etc.

For generality, we also allow entities to have structured forms when necessary.

Then naturally, in such a world with many entities, we would like to partition the state vector h(t) into a set of node blocks {h b (t)} b???B and associate node blocks with entities.

For example, we may associate h mind(alice) (t) to alice and h mind(bob) (t) to bob.

Note that mind(alice) is just an example of the kind of node blocks that can be associated with alice.

There can be another node block associated with the physical condition of alice and be called body(alice).

Of course when there is only one node block associated with alice, we can also simply call it alice.

From now on, we use teal-colored typewriter font for events and orange-colored font for node blocks.

From Figure 1b , we already see that an event may only depend on and affect a subset of hidden nodes in h(t), and this further prompts us to figure out a way to describe our inductive biases on which node blocks are to determine the intensity of a given event as well as which node blocks are to be updated by one.

We propose a general interface based on Datalog-a declarative logic programming language-to assert our inductive biases into a deductive database as facts and rules.

Then as each event happens, we can query the database to figure out which node blocks determine its intensity and which node blocks will be updated by it.

In this section, we walk through our Datalog interface by introducing its keywords one step a time.

We write keywords in boldfaced typewriter font, and color-code them for both presentation and reading convenience.

The colors we use are consistent with the examples in Figure 1 .

We first need to specify what is a legal node block in our system by using the keyword is block:

where b can be replaced with a node block name like alice, bob, chicago and etc.

Such a Datalog statement is a database fact.

Then we use the keyword is event to specify what is a legal event type in our system:

where k can be replaced with email(alice,bob), email(bob,alice), travel(bob,chicago) and etc.

As we may have noticed, there may be many variants of email(S,R) where the variables S and R can take values as alice, bob and etc.

To avoid writing a separate fact for each pair of S and R, we may summarize facts of the same pattern as a rule:

head of rule

body of rule (5a) 3 Similar structured representation of events has been common in natural language semantics (Davidson, 1967) and philosophy (Kim, 1993) .

where :-is used to separate the head and body.

Capitalized identifiers such as S and R denote variables.

A rule mean: for any value of the variables, the head is known to be true if the body is known to be true.

A fact such as is event(email(alice,bob)) is simply a rule with no body (so the :-is omitted), meaning that the body is vacuously true.

To figure out what event types are legal in our system, we can query the database by:

is event(K)?

(6) which returns every event type k that instantiates is event(k).

Note that, unlike a fact or rule that ends with a period (.), a query ends with a question mark (?).

We can declare database rules and facts about which events depend on which node blocks using the depend keyword as:

where k and b are replaced with Datalog variables or values for event and node block respectively, and condition 1 ,...,condition N stands for the body of rule.

An example is as follows:

depend(travel(bob,chicago), X):-resort(X),at(X,chicago).

By querying the database for a given k using depend(k,B)?

we get B d k that is the set of all the node blocks b that instantiates depend(k,b) and has superscript d for depend.

Then we have:

where ??(??) is the sigmoid function, r ranges over all the rules and r depend(k, b) means "the rule r proves the fact depend(k, b)".

The matrices A r ??? R The aggregator ??? represents pooling operation on a set of non-negative vectors.

We choose ??? = and ??? = max because it is appropriate to sum the dependencies over all the rules but extract the "max-dependency" among all the node blocks for each rule.

As shown in equation (8), the intensity of travel(bob,chicago) is determined by both resorts and his friends at chicago so these two possible motivations should be summed up.

But bob may only stay at one of his friends' home and can only afford going to a few places, so only the "best friend" and "signature resort" matter and that is why we use max-pooling for ??? .

As a matter of implementation, we modify each depend rule to have the rule index r as a third argument:

This makes it possible to apply semantics-preserving transformations to the resulting Datalog program without inadvertently changing the neural architecture.

Moreover, if the Datalog programmer specifies the third argument r explicitly, then we do not modify that rule.

As a result, it is possible for multiple rules to share the same r, meaning that they share parameters.

We can declare database rules and facts about which events affect which node blocks using the affect keyword as:

(12) such that we know which node blocks to update as each event happens.

For example, we can allow travel(bob,chicago) to update h X (t) for any X who is a friend of bob and at chicago: affect(travel(bob,chicago), X)):-friend(bob,X), at(X,chicago).

By querying the database for a given k using

we get B a k that is the set of all the node blocks b that instantiates affect(k,b) where the superscript a stands for affect.

Then each node block h b (t) updates itself as shown in equation (2) Similar to how A r and B r in equation (10) are declared, a U r is implicitly declared by each affect rule such that we have:

where ??? = .

This term is analogous to the Uw k term in section 2.1

Note that we can also modify each affect rule (as we do for depend in section 3.2) to have the rule index r as a third argument.

By explicitly specifying r, the Datalog programmer can allow multiple affect rules to share U r .

We can specify how node blocks update one another by using the update keyword:

meaning the node block b updates the node block b when k happens.

Note that b can equal b. It is often useful to write this rule:

which means that whenever K causes B to update, B gets to see its own previous state (as well as K).

To update the node block b with event k, we need

where r ranges over all rules and

This term is analogous to the Vh(t) term in section 2.1.

Having equations (13) and (16), we pass ?? 0,b,k + ?? 1,b,k through the activation functions and obtain the updated h b,new .

Similar to depend and affect, we can also explicitly specify an extra argument r in each update rule to allow multiple rules to share V r .

Parameter sharing (in depend, affect and update) is important because it works as a form of regularization: shared parameters tend to get updated more often than the individual ones, thus leaving the latter less likely to overfit the training data when we "early-stop" the training procedure.

When each event type k is declared using is event(k), the system automatically creates event embedding vectors v k and w k and they will be used in equations (10) and (13) respectively.

When some event types involve many entities which results in a very large number of event types, this design might end up with too many parameters, thus being hard to generalize to unseen data.

We can allow event types to share embedding vectors by adding an extra argument to the keyword is event:

is event(k,m):-condition 1 ,...,condition N .

where m is an index to a pair of embedding vectors v m and w m .

There can be more than one pair that is used by an event type k as shown in this example: is event(email(S,R), S), is event(email(S,R), R), is event(email(S,R), email) and etc.

Then we compute the final embedding vectors of email(S,R) as:

Similar argument in section 3.4 applies here that sharing embedding vectors across event types is a form of regularization.

In a simplified version of our approach, we could use a homogeneous neural architecture where all events have the same dimension, etc.

In our actual implementation, we allow further flexibility by using Datalog rules to define dimensionalities, activation functions, and multi-layer structures for event embeddings.

This software design is easy to work with, but is orthogonal to the machine learning contribution of the paper, so we describe it in Appendix A.4.

Learning Following Mei & Eisner (2017), we can learn the parameters of the proposed model by locally maximizing in equation (19) using any stochastic gradient method: Its log-likelihood given the sequence over the observation interval [0, T ] is as follows:

The only difference is that our Datalog program affects the neural architecture, primarily by dictating that some weights in the model are structurally zero.

Concretely, to compute and its gradient, as each event e i = k i @t i happens, we need to query the database with depend(k,B)?

for the node blocks that each k depends on in order to compute log ?? ki (t i ) and the Monte Carlo approximation to

Then we need to query the database with affect(k,B)?

for the node blocks to be affected and update them.

A detailed recipe is Algorithm 1 of Appendix B.1 including a down-sampling trick to handle large K.

Prediction Given an event sequence prefix k 1 @t 1 , k 2 @t 2 , . . .

, k i???1 @t i???1 , we may wish to predict the time and type of the next event.

The time t i has density p i (t) = ??(t) exp ??? t ti???1 ??(s)ds where ??(t) = k???K ?? k (t), and we choose ??? ti???1 tp i (t)dt as the time prediction because it has the lowest expected L 2 loss.

Given the next event time t i , the most likely type would simply be arg max k ?? k (t i ), but the most likely next event type without knowledge of t i is arg max k

The integrals in the preceding equations can be estimated using i.i.d.

samples of t i drawn from p i (t).

We draw t i using the thinning algorithm (Lewis & Shedler, 1979; Liniger, 2009; Mei & Eisner, 2017) .

Given t i , we draw k i from the distribution where the probability of each type k is proportional to ?? k (t i ).

A full sequence can be rolled out by repeatedly feeding the sampled event back into the model and then drawing the next.

See Appendix B.2 for implementation details.

We show how to use our Datalog interface to inject inductive biases into the neural Hawkes process (NHP) on multiple synthetic and real-world datasets.

On each dataset, we compare the model with modified architecture-we call it structured neural Hawkes process (or structured-NHP) with the plain vanilla NHP on multiple evaluation metrics.

See Appendix C for experimental details (e.g., dataset statistics and training details).

We implemented the model in PyTorch (Paszke et al., 2017 ).

As Mei & Eisner (2017) pointed out, it is important for a model family to handle the superposition of real-time sequences, because in various real settings, some event types tend not to interact.

For example, the activities of two strangers rarely influence each other, although they are simultaneously monitored and thus form a single observed sequence.

In this section, we experiment on the data known to be drawn from a superposition of M neural Hawkes processes with randomly initialized parameters.

Each process X has four event types event(K,X) where K can be 1, 2, 3 and 4.

To leverage the knowledge about the superposition structure, one has to either implement a mixture of neural Hawkes processes or transform a single neural Hawkes process to a superposition model by (a) zeroing out specific elements of v k such that ?? k (t) for k ??? K X depends on only a subset S of the LSTM hidden nodes, (b) setting specific LSTM parameters such that events of type k ??? K Y don't affect the nodes in S and (c) making the LSTM transition matrix a blocked-structured matrix such that different node blocks don't update each other.

Neither way is trivial.

With our Datalog interface, we can explicitly construct such a superposition process rather easily by writing simple datalog rules as follows:

(20b) update(X, unit(X)):-is block(X).

Events of X do not influence Y at all, and processes don't share parameters.

We generated learning curves (Figure 2 ) by training a structured-NHP and a NHP on increasingly long prefixes of the training set.

As we can see, the structured model substantially outperform NHP at all training sizes.

The neural Hawkes process gradually improves its performance as more training sequences become available: it perhaps learns to set its w k and LSTM parameters from data.

However, thanks to the right inductive bias, the structured model requires much less data to achieve somewhat close to the oracle performance.

Actually, as shown in Figure 2 , the structured model only needs 1/16 of training data as NHP does to achieve a higher likelihood.

The improvement of the structured model over NHP is statistically significant with p-value < 0.01 as shown by the pair-permutation test at all training sizes of all the datasets.

Elevator System Dataset (Crites & Barto, 1996) .

In this dataset, two elevator cars transport passengers across five floors in a building (Lewis, 1991; Bao et al., 1994; Crites & Barto, 1996) .

Each event type has the form stop(C,F) meaning that C stops at F to pick up or drop off passengers where C can be car1 and car2 and F can be floor1, . . .

, floor5.

This dataset is representative of many real-world domains where individuals physically move from one place to another for, e.g., traveling, job changing, etc.

With our Datalog interface, we can explicitly express our inductive bias that each stop(C,F) depends on and affects the associated node blocks C and F:

(21b)

The set of inductive biases is desirable because whether a C will head to a F and stops there is primarily determined by C's state (e.g., whether it is already on the way of sending anyone to that floor) and F's state (e.g., whether there is anyone on that floor waiting for a car).

We also declare a global node block, building, that depends on and affects every event in order to compensate for any missing knowledge (e.g., the state of the joint controller for the elevator bank, and whether it's a busy period for the humans) and/or missing data (e.g., passengers arrive at certain floors and press the buttons).

Appendix C.2 gives a full Datalog specification of the model that we used for the experiments in this domain.

More details about this dataset (e.g. pre-processing) can be found in Appendix C.1.2.

EuroEmail Dataset (Paranjape et al., 2017) .

In this domain, we model the email communications between anonymous members of an European research institute.

Each event type has the form email(S,R) meaning that S sends an email to R where S and R are variables that take the actual members as values.

With our Datalog interface, we can express our knowledge that each event depends on and affects its sender and receiver as the following rules:

depend(send(S,R), S).

depend(send(S,R), R).

Appendix C.2 gives a full Datalog specification of the model that we used for the experiments in this domain.

More details about this dataset (e.g. pre-processing) can be found in Appendix C.1.3.

We evaluate the models in three ways as shown in Figure 3 .

We first plot learning curves (Figure 3a) by training a structured-NHP and an NHP on increasingly long prefixes of each training set.

Then we show the per-sequence scatterplots in Figure 3b .

We can see that either in learning curve or scatterplots, structured-NHP consistently outperforms NHP, which proves that structured-NHP is both more data-efficient and more predictive.

Finally, we compare the models on the prediction tasks and datasets as shown in Figure 3c .

We make minimum Bayes risk predictions as explained in section 4.

We evaluate the type prediction with 0-1 loss, yielding an error rate.

We can see, in both of Elevator and EuroEmail datasets, structured-NHP could be significantly more accurate on type prediction.

We evaluate the time prediction with L 2 loss, and reported the mean squared error as a percentage of the variance of the true time interval (denoted as MSE%).

Note that we get can MSE%=1.0 if we always predict t i as t i???1 + ???t where ???t is the average length of time intervals.

Figure 3c shows that the structured model outperforms NHP on event type prediction on both datasets, although for time prediction they perform neck to neck.

We speculate that it might be because the structure information is more directly related to the event type (because of its structured term) but not time.

There has been extensive research about having inductive biases in the architecture design of a machine learning model.

The epitome of this direction is perhaps the graphical models where edges between variables are usually explicitly allowed or forbidden (Koller & Friedman, 2009 ).

There has also been work in learning such biases from data.

For example, Stepleton et al. (2009) proposed to encourage the block-structured states for Hidden Markov Models (HMM) by enforcing a sparsityinducing prior over the non-parametric Bayesian model.

Duvenaud et al. (2013) and Brati??res et al. (2014) attempted to learn structured kernels for Gaussian processes.

Our work is in the direction of injecting inductive biases into a neural temporal model-a class of models that is useful in various domains such as demand forecasting (Seeger et al., 2016) , personalization and recommendation (Jing & Smola, 2017) , event prediction (Du et al., 2016) and knowledge graph modeling (Trivedi et al., 2017) .

Incorporating structural knowledge in the architecture design of such a model has drawn increasing attention over the past few years.

Shelton & Ciardo (2014) introduced a factored state space in continuous-time Markov processes.

Meek (2014) and Bhattacharjya et al. (2018) proposed to consider direct dependencies among events in graphical event models.

Wang et al. (2019) developed a hybrid model that decomposes exchangeable sequences into a global part that is associated with common patterns and a local part that reflects individual characteristics.

However, their approaches are all bounded to the kinds of inductive biases that are easy to specify (e.g. by hand).

Our work enables people to use a Datalog program to conveniently specify the neural architecture based on a deductive database-a much richer class of knowledge than the previous work could handle.

Although logic programming languages and databases have both previously been used to derive a graphical model architecture (Getoor & Taskar, 2007) , we are, to the best of our knowledge, the first to develop such a general interface for a neural event model.

As future work, we hope to develop an extension where events can also trigger assertions and retractions of facts in the Datalog database.

Thanks to the Datalog rules, the model architecture will dynamically change along with the facts.

For example, if Yoyodyne Corp. hires Alice, then the Yoyodyne node block begins to influence Alice's actions, and K expands to include a new (previously impossible) event where Yoyodyne fires Alice.

Moreover, propositions in the database-including those derived via other Datalog rules-can now serve as extra bits of system state that help define the ?? k intensity functions in (1).

Then the system's learned neural state s i is usefully augmented by a large, exact set of boolean propositions-a division of labor between learning and expert knowledge.

In this section, we elaborate on the details of the transition function ?? that is introduced in section 2.1; more details about them may be found in Mei & Eisner (2017) .

where the interval (t i???1 , t i ] has consecutive observations k i???1 @t i???1 and k i @t i as endpoints.

At t i , the continuous-time LSTM reads k i @t i and updates the current (decayed) hidden cells c(t) to new initial values c i+1 , based on the current (decayed) hidden state h(t i ), as follows:

At time t i , the updated state vector is

] is given by (26), which continues to control h(t) except that i has now increased by 1).

On the interval (t i , t i+1 ], c(t) follows an exponential curve that begins at c i+1 (in the sense that lim t???t + i c(t) = c i+1 ) and decays, as time t increases, toward c i+1 (which it would approach as t ??? ???, if extrapolated).

We initialize each node block h b (0) = 0, and then have it read a special beginning-of-stream (BOS) event bos@t 0 where bos is a special event type and t 0 is set to be 0.

Then equations (24)- (25) define c 1 (from c 0 def = 0), c 1 , ?? 1 , and o 1 .

This is the initial configuration of the system as it waits for the first event to happen: this initial configuration determines the hidden state h(t) and the intensity functions ?? k (t) over t ??? (0, t 1 ].

The bos event affects every node block but depends on none of them because we do not generate it.

When the system is initiated, the following rule is automatically asserted by our program so users don't have to do it by themselves.

affect(bos,X):-is block(X).

More details about why bos is desirable can be found in Mei & Eisner (2017) .

The vanilla neural Hawkes process can be specified using our interface as follows:

(28b) update(global,global,K).

(28c) where h global (t) is the only node block that every event type k depends on and affects.

Equation (10) falls back to f k (v k ??(A??(BCh global (t)))) which is not exactly the same with, yet at least as expressive as equation (1).

A.4 OPTIONAL architecture, input AND output KEYWORDS As discussed in section 3.5, the embedding vector of each event is just the sum of trainable vectors.

Actually, we further allow users to write Datalog rules to define embedding models that have multilayer structures and activation functions of interest.

We can define a L-layer neural network using the architecture keyword as:

where n is a (structured) term as the model name, D 0 is the input dimension, D l and a l are the output dimension and activation type of l-th layer respectively.

The example below defines a model named emb that has a neural layer with hyper-tangent activation followed by a linear layer.

architecture (

Note that we allow using = for architecture to indicate that there should be only one model under each name n, although it is not supported in the standard datalog implementation.

We can assign to each k a model n and spell out its arguments x 1 , x 2 , . . . (to be concatenated in order) for input embedding computation using the input keyword:

and follow the same format for output embedding computation using the output keyword.

Note that we use = again.

The example below means that each w email(S,R) is computed by passing the concatenation of S and R into model emb and that v email(S,R) is computed the same way:

input(email(S,R))= emb(S,R).

(32a) output(email(S,R))= emb(S,R).

B ALGORITHM DETAILS

In this section, we elaborate on the details of algorithms.

The log-likelihood in equation (19) can be computed by calling Algorithm 1.

The down sampling trick (line 32 of Algorithm 1) can be used when there are too many event types.

It gives an unbiased estimate of the total intensity k???K ?? k (t), yet remains much less computationally expensive especially when J |K|.

In our experiments, we found that its variance over the entire corpus turned out small, although it may, in theory, suffer large variance.

As future work, we will explore sampling from proposal distributions where the probability of choosing any k is (perhaps trained to be) proportional to its actual intensity ?? k (t), in order to further reduce the variance.

But this is not within the scope of this paper.

Note that, in principle, we have to make Datalog queries after every event, to figure out which node blocks are affected by that event and to find the new intensities of all events.

However, certain Datalog queries may be slow.

Thus, in practice, rather than repeatedly making the same queries, we just memorize the result the first time and look it up when it is needed again.

Problems emerge when events are allowed to change the database (e.g. asserting and retracting facts as in Appendix D), then this may change the results of some queries, and thus the memos for those queries are now incorrect.

In this case, we might explore using some other more flexible query language that creates memos and keeps them up to date (Filardo & Eisner, 2012) .

Given an event sequence prefix k 1 @t 1 , k 2 @t 2 , . . . , k i???1 @t i???1 , we can call Algorithm 2 to draw the single next event.

A full sequence can be rolled out by repeatedly feeding the sampled event back into the model and then drawing the next (calling Algorithm 2 another time).

How do we construct the upper bound ?? * (line 8 of Algorithm 2)?

We express the upper bound as ?? * = k???K ?? * k and find ?? * k ??? ?? k (t) for each k. We copy the formulation of ?? k (t) here for easy reference:

where each summand g dd h bd (t) = g dd ??o id ??(2??(2c d (t))??? 1) is upper-bounded by max c???{c id ,c id } g dd ?? o id ?? (2??(2c) ??? 1).

Note that the coefficients g dd may be either positive or negative.

C EXPERIMENTAL DETAILS C.1 DATASET STATISTICS Table 1 shows statistics about each dataset that we use in this paper.

We synthesize data by sampling event sequences from different structured processes.

Each structured process is a mixture model of M neural Hawkes processes and each neural Hawkes process(X) has four event types event1(X), event2(X), event3(X) and event4(X).

We chose M = 4, 8, 16 and end up with three different datasets.

We chose the sequence length I = 21 and then used the thinning algorithm (Lewis & Shedler, 1979; Liniger, 2009; Mei & Eisner, 2017) to sample the first I events over [0, ???).

We set T = t I , i.e., the time of the last generated event.

We generate 2000, 100 and 100 sequences for each training, dev, and test set respectively.

We examined our method in a simulated 5-floor building with 2 elevator cars.

The system was initially built in Fortran by Crites & Barto (1996) and then rebuilt in Python by Mei et al. (2019) .

During a typical afternoon down-peak rush hour (when passengers go from floor-2,3,4,5 down to the lobby), elevator cars travel to each floor and pick up passengers that have (stochastically) arrived there according to a traffic profile that can be found in (Bao et al., 1994) and Mei et al. (2019) .

In this dataset, each event type is stop(C,F) where C can be car1 and car2 and F can be floor1, . . . , floor5.

So there are 10 event types in total in this simulated building.

We repeated the (one-hour) simulation 1200 times to collect the event sequences, each of which has around 1200 time-stamped records of which car stops at which floor.

We randomly sampled disjoint train, dev and test sets with 1000, 100 and 100 sequences respectively.

EuroEmail is proposed by Paranjape et al. (2017) .

It was generated using email data from a large European research institute, and was highly anonymized.

The emails only represent communications between institution members, which are indexed by integers, with timestamps.

In the dataset are 986 users and 332334 email communications spanning over 800 days.

However, most users only send or receive one or two emails, leaving this dataset extremely sparse.

We extracted all the emails among the top 20 most active users, and end up with 5881 emails.

We split the single long sequence into 120 sequences with average length of 48, and set the training, dev, test size as 100, 10, 10 respectively.

In this dataset, event type is defined as send(S,R), where S and R are members in this organization.

Then there're 20 ?? 20 = 400 different event types, where we assume that people may send emails to themselves.

In this section, we give a full Datalog specification of the model that we used for the experiments on each dataset.

Here is the full program for Elevator domain.

(33a) is block(car2).

(33b) is block(floor1).

(33c) is block(floor2).

(33d) is block(floor3).

(33e) is block(floor4).

(33f) is block(floor5).

(33g) is block(building).

(33h) is car(car1).

(33i) is car(car2).

(33j) is floor(floor1).

(33k) is floor(floor2).

(33l) is floor(floor3).

(33m) is floor(floor4).

(33n) is floor(floor5).

(33o) is event(stop(C,F)):-is car(C),is floor(F).

(33p) depend(stop(C,F), C).

(33q) depend(stop(C,F), F).

(33r) depend(stop(C,F), building).

(33s)

@highlight

Factorize LSTM states and zero-out/tie LSTM weight matrices according to real-world structural biases expressed by Datalog programs.