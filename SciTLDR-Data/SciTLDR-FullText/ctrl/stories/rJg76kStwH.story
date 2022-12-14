Markov Logic Networks (MLNs), which elegantly combine logic rules and probabilistic graphical models, can be used to address many knowledge graph problems.

However, inference in MLN is computationally intensive, making the industrial-scale application of MLN very difficult.

In recent years, graph neural networks (GNNs) have emerged as efficient and effective tools for large-scale graph problems.

Nevertheless, GNNs do not explicitly incorporate prior logic rules into the models, and may require many labeled examples for a target task.

In this paper, we explore the combination of MLNs and GNNs, and use graph neural networks for variational inference in MLN.

We propose a GNN variant, named ExpressGNN, which strikes a nice balance between the representation power and the simplicity of the model.

Our extensive experiments on several benchmark datasets demonstrate that ExpressGNN leads to effective and efficient probabilistic logic reasoning.

Knowledge graphs collect and organize relations and attributes about entities, which are playing an increasingly important role in many applications, including question answering and information retrieval.

Since knowledge graphs may contain incorrect, incomplete or duplicated records, additional processing such as link prediction, attribute classification, and record de-duplication is typically needed to improve the quality of knowledge graphs and derive new facts.

Markov Logic Networks (MLNs) were proposed to combine hard logic rules and probabilistic graphical models, which can be applied to various tasks on knowledge graphs (Richardson & Domingos, 2006) .

The logic rules incorporate prior knowledge and allow MLNs to generalize in tasks with small amount of labeled data, while the graphical model formalism provides a principled framework for dealing with uncertainty in data.

However, inference in MLN is computationally intensive, typically exponential in the number of entities, limiting the real-world application of MLN.

Graph neural networks (GNNs) have recently gained increasing popularity for addressing many graph related problems effectively (Dai et al., 2016; Li et al., 2016; Kipf & Welling, 2017; Schlichtkrull et al., 2018) .

However, the design and training procedure of GNNs do not explicitly take into account the prior knowledge in the form of logic rules.

To achieve good performance, these models typically require sufficient labeled instances on specific end tasks (Xiong et al., 2018) .

In this paper, we explore the combination of the best of both worlds, aiming for a method which is data-driven yet can exploit the prior knowledge encoded in logic rules.

To this end, we design a simple variant of graph neural networks, named ExpressGNN, which can be efficiently trained in the variational EM framework for MLN.

An overview of our method is illustrated in Fig. 1 .

ExpressGNN and the corresponding reasoning framework lead to the following desiderata:

??? Efficient inference and learning: ExpressGNN can be viewed as the inference network for MLN, which scales up MLN inference to much larger knowledge graph problems.

??? Combining logic rules and data supervision: ExpressGNN can leverage the prior knowledge encoded in logic rules, as well as the supervision from labeled data.

??? Compact and expressive model: ExpressGNN may have small number of parameters, yet it is sufficient to represent mean-field distributions in MLN.

Statistical relational learning.

There is an extensive literature relating the topic of logic reasoning, and here we only focus on the approaches that are most relevant to statistical relational learning on knowledge graphs.

Logic rules can compactly encode the domain knowledge and complex dependencies.

Thus, hard logic rules are widely used for reasoning in earlier attempts, such as expert systems (Ignizio, 1991) and inductive logic programming (Muggleton & De Raedt, 1994) .

However, hard logic is very brittle and has difficulty in coping with uncertainty in both the logic rules and the facts in knowledge graphs.

Later studies have explored to introduce probabilistic graphical model in the field of logic reasoning, seeking to combine the advantages of relational and probabilistic approaches.

Representative works including Relational Markov Networks (RMNs; Taskar et al. (2007) ) and Markov Logic Networks (MLNs; Richardson & Domingos (2006) ) were proposed in this background.

Markov Logic Networks.

MLNs have been widely studied due to the principled probabilistic model and effectiveness in a variety of reasoning tasks, including entity resolution (Singla & Domingos, 2006a) , social networks (Zhang et al., 2014) , information extraction (Poon & Domingos, 2007) , etc.

MLNs elegantly handle the noise in both logic rules and knowledge graphs.

However, the inference and learning in MLNs is computationally expensive due to the exponential cost of constructing the ground Markov network and the NP-complete optimization problem.

This hinders MLNs to be applied to industry-scale applications.

Many works appear in the literature to improve the original MLNs in both accuracy (Singla & Domingos, 2005; Mihalkova & Mooney, 2007) and efficiency (Singla & Domingos, 2006b; Poon & Domingos, 2006; Khot et al., 2011; Bach et al., 2015) .

Nevertheless, to date, MLNs still struggle to handle large-scale knowledge bases in practice.

Our framework ExpressGNN overcomes the scalability challenge of MLNs by efficient stochastic training algorithm and compact posterior parameterization with graph neural networks.

Graph neural networks.

Graph neural networks (GNNs; Dai et al. (2016) ; Kipf & Welling (2017) ) can learn effective representations of nodes by encoding local graph structures and node attributes.

Due to the compactness of model and the capability of inductive learning, GNNs are widely used in modeling relational data (Schlichtkrull et al., 2018; Battaglia et al., 2018) .

Recently, proposed Graph Markov Neural Networks (GMNNs), which employs GNNs together with conditional random fields to learn object representations.

These existing works are simply data-driven, and not able to leverage the domain knowledge or human prior encoded in logic rules.

To the best of our knowledge, ExpressGNN is the first work that connects GNNs with first-order logic rules to combine the advantages of both worlds.

Knowledge graph embedding.

Another line of research for knowledge graph reasoning is in the family of knowledge graph embedding methods, such as TransE (Bordes et al., 2013) , NTN (Socher et al., 2013) , DistMult (Kadlec et al., 2017) , ComplEx (Trouillon et al., 2016) , and RotatE (Sun et al., 2019) .

These methods design various scoring functions to model relational patterns for knowledge graph reasoning, which are very effective in learning the transductive embeddings of both entities and relations.

However, these methods are not able to leverage logic rules, which can be crucial in some relational learning tasks, and have no consistent probabilistic model.

Compared to these methods, ExpressGNN has consistent probabilistic model built in the framework, and can incorporate knowledge from logic rules.

A recent concurrent work has proposed probabilistic Logic Neural Network (pLogicNet), which integrates knowledge graph embedding methods with MLNs with EM framework.

Compared to pLogicNet which uses a flattened embedding table as the entity representation, our work can better capture the structure knowledge encoded in the knowledge graph with GNNs, and supplement the knowledge from logic formulae for the prediction task.

Our method is a general framework that can trade-off the model compactness and expressiveness by tuning the dimensionality of the GNN part and the embedding part.

Thus, pLogicNet can be viewed as a special case of our work with the embedding part only.

3 PRELIMINARY Knowledge Graph.

A knowledge graph is a tuple

In the language of first-order logic, entities are also called constants.

For instance, a constant can be a person or an object.

Relations are also called predicates.

Each predicate is a logic function defined over C, i.e., r(??) : C ?? . . .

?? C ??? {0, 1} .

In general, the arguments of predicates are asymmetric.

For instance, for the predicate r(c, c ) := L(c, c ) (L for Like) which checks whether c likes c , the arguments c and c are not exchangeable.

With a particular set of entities assigned to the arguments, the predicate is called a ground predicate, and each ground predicate ??? a binary random variable, which will be used to define MLN.

For a d-ary predicate, there are M d ways to ground it.

We denote an assignment as a r .

For instance, with a r = (c, c ), we can simply write a ground predicate r(c, c ) as r(a r ).

Each observed fact in knowledge bases is a truth value {0, 1} assigned to a ground predicate.

For instance, a fact o can be [L(c, c ) = 1].

The number of observed facts is typically much smaller than that of unobserved facts.

We adopt the open-world paradigm and treat these unobserved facts ??? latent variables.

As a clearer representation, we express a knowledge base K by a bipartite graph G K = (C, O, E), where nodes on one side of the graph correspond to constants C and nodes on the other side correspond to observed facts O, which is called factor in this case.

The set of T edges, E = {e 1 , . . . , e T }, connect constants and the observed facts.

More specifically, an edge e = (c, o, i) between node c and o exists, if the ground predicate associated with o uses c as an argument in its i-th argument position (Fig. 2) .

Markov Logic Networks.

MLNs use logic formulae to define potential functions in undirected graphical models.

A logic formula f (??) : C ?? . . .

?? C ??? {0, 1} is a binary function defined via the composition of a few predicates.

For instance, a logic formula f (c, c ) can be

where ?? is negation and the equivalence is established by De Morgan's law.

Similar to predicates, we denote an assignment of constants to the arguments of a formula f as a f , and the entire collection of consistent assignments of constants as A f = {a 1 f , a 2 f , . . .}.

A formula with constants assigned to all of its arguments is called a ground formula.

Given these logic representations, MLN can be defined as a joint distribution over all observed facts O and unobserved facts H as

where Z(w) is the partition function summing over all ground predicates and ?? f (??) is the potential function defined by a formula f as illustrated in Fig. 2 .

One form of ?? f (??) can simply be the truth value of the logic formula f .

For instance, if the formula is f (c, c ) := ??S(c) ??? ??F(c, c ) ??? S(c ), then ?? f (c, c ) can simply take value 1 when f (c, c ) is true and 0 otherwise.

Other more sophisticated ?? f can also be designed, which have the potential to take into account complex entities, such as images or texts, but will not be the focus of this paper.

The weight w f can be viewed as the confidence score of the formula f : the higher the weight, the more accurate the formula is.

Difference between KG and MLN.

We note that the graph topology of knowledge graphs and MLN can are very different, although MLN is defined on top of knowledge graphs.

Knowledge graphs are typically very sparse, where the number of edges (observed relations) is typically linear in the number of entities.

However, the graphs associated with MLN are much denser, where the number of nodes can be quadratic or more in the number of entities, and the number of edges (dependency between variables) is also high-order polynomials in the number of entities.

In this section, we introduce the variational EM framework for MLN inference and learning, where we will use ExpressGNN as a key component (detailed in Sec. 5).

Markov Logic Networks model the joint probabilistic distribution of all observed and latent variables, as defined in Eq. 1.

This model can be trained by maximizing the log-likelihood of all the observed facts log P w (O).

However, it is intractable to directly maximize the objective, since it requires to compute the partition function Z(w) and integrate over all variables O and H. We instead optimize the variational evidence lower bound (ELBO) of the data log-likelihood, as follows

where Q ?? (H | O) is a variational posterior distribution of the latent variables given the observed ones.

The equality in Eq. 2 holds if the variational posterior Q ?? (H|O) equals to the true posterior P w (H|O).

We then use the variational EM algorithm (Ghahramani et al., 2000) to effectively optimize the ELBO.

The variational EM algorithm consists of an expectation step (E-step) and a maximization step (M-step), which will be called in an alternating fashion to train the model: 1) In the E-step (Sec. 4.1), we infer the posterior distribution of the latent variables, where P w is fixed and Q ?? is optimized to minimize the KL divergence between Q ?? (H|O) and P w (H|O); 2) In the M-step (Sec. 4.2), we learn the weights of the logic formulae in MLN, where Q ?? is fixed and P w is optimized to maximize the data log-likelihood.

In the E-step, which is also known as the inference step, we are minimizing the KL divergence between the variational posterior distribution Q ?? (H|O) and the true posterior distribution P w (H|O).

The exact inference of MLN is computationally intractable and proven to be NP-complete (Richardson & Domingos, 2006) .

Therefore, we choose to approximate the true posterior with a mean-field distribution, since the mean-field approximation has been demonstrated to scale up large graphical models, such as latent Dirichlet allocation for modeling topics from large text corpus (Hoffman et al., 2013) .

In the mean-field variational distribution, each unobserved ground predicate r(a r ) ??? H is independently inferred as follows:

where each factorized distribution Q ?? (r(a r )) follows the Bernoulli distribution.

We parameterize the variational posterior Q ?? with deep learning models as our neural inference network.

The design of the inference network is very important and has a lot of considerations, since we need a compact yet expressive model to accurately approximate the true posterior distribution.

We employ graph neural networks with tunable embeddings as our inference network (detailed in Sec. 5), which can trade-off between the model compactness and expressiveness.

With the mean-field approximation, L ELBO (Q ?? , P w ) defined in Eq. 2 can be reorganized as below:

where w f is fixed in the E-step and thus the partition function Z(w) can be treated as a constant.

We notice that the first term E Q ?? (H|O) [log P w (O, H)] has the summation over all formulae and all possible assignments to each formula.

Thus this double summation may involve a large number of terms.

The second term E Q ?? (H|O) [log Q ?? (H|O)] is the sum of entropy of the variational posterior distributions Q ?? (r(a r )), which also involves a large number of terms since the summation ranges over all possible latent variables.

Typically, the number of latent facts in database is much larger than the number of observed facts.

Thus, both terms in the objective function pose the challenge of intractable computational cost.

To address this challenge, we sample mini-batches of ground formulae to break down the exponential summations by approximating it with a sequence of summations with a controllable number of terms.

More specifically, in each optimization iteration, we first sample a batch of ground formulae.

For each ground formula in the sampled batch, we compute the first term in Eq. 4 by taking the expectation of the corresponding potential function with respect to the posterior of the involved latent variables.

The mean-field approximation enables us to decompose the global expectation over the entire MLN into local expectations over ground formulae.

Similarly, for the second term in Eq. 4, we use the posterior of the latent variables in the sampled batch to compute a local sum of entropy.

For tasks that have sufficient labeled data as supervision, we can add a supervised learning objective to enhance the inference network, as follows:

This objective is complementary to the ELBO on predicates that are not well covered by logic rules but have sufficient observed facts.

Therefore, the overall E-step objective function becomes:

where ?? is a hyperparameter to control the weight.

This overall objective essentially combines the knowledge in logic rules and the supervision from labeled data.

In the M-step, which is also known as the learning step, we are learning the weights of logic formulae in Markov Logic Networks with the variational posterior Q ?? (H|O) fixed.

The partition function Z(w) in Eq. 4 is not a constant anymore, since we need to optimize those weights in the M-step.

There are exponential number of terms in the partition function Z(w), which makes it intractable to directly optimize the ELBO.

To tackle this problem, we adopt the widely used pseudo-log-likelihood (Richardson & Domingos, 2006) as an alternative objective for optimization, which is defined as:

where MB r(ar) is the Markov blanket of the ground predicate r(a r ), i.e., the set of ground predicates that appear in some grounding of a formula with r(a r ).

For each formula i that connects r(a r ) to its Markov blanket, we optimize the formula weight w i by gradient descent, with the derivative:

where y r(ar) = 0 or 1 if r(a r ) is an observed fact, and y r(ar) = Q ?? (r(a r )) otherwise.

With the independence property of Markov Logic Networks, the gradients of the logic formulae weights can be efficiently computed on the Markov blanket of each variable.

For the M-step, we design a different sampling scheme to make it computationally efficient.

For each variable in the Markov blanket, we take the truth value if it's observed and draw a sample from the variational posterior Q ?? if it's latent.

In the M-step, the ELBO of a fully observed ground formula depends on the formula weight, thus we need to consider all the fully observed ground formulae.

It is computationally intractable to use all possible ground predicates to compute the gradients in Eq. 8.

To tackle this challenge, we simply consider all the ground formulae with at most one latent predicate, and pick up the ground predicate if its truth value determines the formula's truth value.

Therefore, we keep a small subset of ground predicates, each of which can directly determine the truth value of a ground formula.

Intuitively, this small subset contains all representative ground predicates, and makes good estimation of the gradients with much cheaper computational cost.

Algorithm 1: GNN() Initialize entity node:

In the neural variational EM framework, the key component is the posterior model, or the inference network.

We need to design the inference network that is both expressive and efficient to approximate the true posterior distribution.

A recent concurrent work uses a flattened embedding table as the entity representation to model the posterior.

However, such simple posterior model is not able to capture the structure knowledge encoded in the knowledge graph.

We employ graph neural networks with tunable embeddings to design our inference network.

We also investigate the expressive power of GNN from theoretical perspective, which justifies our design.

Our inference network, named ExpressGNN, consists of three parts: the first part is a vanilla graph neural network (GNN), the second part uses tunable embeddings, and the third part uses the embeddings to define the variational posterior.

For simplicity, we assume that each predicate has two arguments (i.e., consider only r(c, c )).

We design each part as follows:

??? We build a GNN on the knowledge graph G K , which is much smaller than the ground graph of MLN (see comparison in Fig. 2 ).

The computational graph of the GNN is given in Algorithm 1.

The GNN parameters ?? 1 and ?? 2 are shared across the entire graph and independent of the number of entities.

Therefore, the GNN is a compact model with

??? For each entity in the knowledge graph, we augment its GNN embedding with a tunable embedding

The tunable embeddings increase the expressiveness of the model.

As there are M entities, the number of parameters in tunable embeddings is O(kM ).

??? We use the augmented embeddings of c 1 and c 2 to define the variational posterior.

Specifically,

In summary, ExpressGNN can be viewed as a two-level encoding of the entities: the compact GNN assigns similar embeddings to similar entities in the knowledge graph, while the expressive tunable embeddings provide additional model capacity to encode entity-specific information beyond graph structures.

The overall number of trainable parameters in ExpressGNN is O(d 2 + kM ).

By tuning the embedding size d and k, ExpressGNN can trade-off between the model compactness and expressiveness.

For large-scale problems with a large number of entities (M is large), ExpressGNN can save a lot of parameters by reducing k.

The combination of GNN and tunable embeddings makes the model sufficiently expressive to approximate the true posterior distributions.

Here we provide theoretical analysis on the expressive power of GNN in the mean-field inference problem, and discuss the benefit of combining GNN and tunable embeddings in ExpressGNN.

Recent studies (Shervashidze et al., 2011; Xu et al., 2018) show that the vanilla GNN embeddings can represent the results of graph coloring, but fail to represent the results of the more strict graph isomorphism check, i.e., GNN produces the same embedding for some nodes that should be distinguished.

We first demonstrate this problem by a simple example:

Example.

??? Entity A and B have opposite relations with E, i.e., F(A, E) = 1 versus F(B, E) = 0 in the knowledge graph, but running GNN on the knowledge graph will always produce the same embeddings for A and B, i.e., ?? A = ?? B .

??? L(A, E) and L(B, E) apparently have different posteriors.

However, using GNN em-

We can formally prove that solving the problem in the above example requires the graph embeddings to distinguish any non-isomorphic nodes in the knowledge graph.

A formal statement is provided below (see Appendix E for the proof).

Definition 5.1.

Two ordered sequences of nodes (c 1 , . . . , c n ) and (c 1 , . . . , c n ) are isomorphic in a graph G K if there exists an isomorphism ?? : Implied by the theorem, to obtain an expressive enough representation for the posterior, we need a more powerful GNN variant.

A recent work has proposed a powerful GNN variant (Maron et al., 2019) , which can handle small graphs such as chemical compounds and protein structures, but it is computationally expensive due to the usage of high-dimensional tensors.

As a simple yet effective solution, ExpressGNN augments the vanilla GNN with additional tunable embeddings, which is a trade-off between the compactness and expressiveness of the model.

In summary, ExpressGNN has the following nice properties:

??? Efficiency: ExpressGNN directly works on the knowledge graph, instead of the huge MLN grounding graph, making it much more efficient than the existing MLN inference methods.

??? Compactness:

The compact GNN model with shared parameters can be very memory efficient, making ExpressGNN possible to handle industry-scale problems.

??? Expressiveness: The GNN model can capture structure knowledge encoded in the knowledge graph.

Meanwhile, the tunable embeddings can encode entity-specific information, which compensates for GNN's deficiency in distinguishing non-isomorphic nodes.

??? Generalizability: With the GNN embeddings, ExpressGNN may generalize to new entities or even different but related knowledge graphs unseen during training time without the need for retraining.

Benchmark datasets.

We evaluate ExpressGNN and other baseline methods on four benchmark datasets: UW-CSE (Richardson & Domingos, 2006) , Cora (Singla & Domingos, 2005) , synthetic Kinship datasets, and FB15K-237 (Toutanova & Chen, 2015) constructed from Freebase (Bollacker et al., 2008) .

Details and full statistics of the benchmark datasets are provided in Appendix B.

General settings.

We conduct all the experiments on a GPU-enabled (Nvidia RTX 2080 Ti) Linux machine powered by Intel Xeon Silver 4116 processors at 2.10GHz with 256GB RAM.

We implement ExpressGNN using PyTorch and train it with Adam optimizer (Kingma & Ba, 2014) .

To ensure a fair comparison, we allocate the same computational resources (CPU, GPU and memory) for all the experiments.

We use the default tuned hyperparameters for competitor methods, which can reproduce the experimental results reported in their original works.

Model hyperparameters.

For ExpressGNN, we use 0.0005 as the initial learning rate, and decay it by half for every 10 epochs without improvement of validation loss.

For Kinship, UW-CSE and Cora, we run ExpressGNN with a fixed number of iterations, and use the smallest subset from the original split for hyperparameter tuning.

For FB15K-237, we use the original validation set to tune the hyperparameters.

We use a two-layer MLP with ReLU activation function as the nonlinear transformation for each embedding update step in the GNN model.

We learn different MLP For each dataset, we search the configuration of ExpressGNN on either the validation set or the smallest subset.

The configuration we search includes the embedding size, the split point of tunable embeddings and GNN embeddings, the number of embedding update steps, and the sampling batch size.

For the inference experiments, the weights for all the logic formulae are fixed as 1.

For the learning experiments, the weights are initialized as 1.

For the choice of ?? in the combined objective L ?? in Eq. 6, we set ?? = 0 for the inference experiments, since the query predicates are never seen in the training data and no supervision is available.

For the learning experiments, we set ?? = 1.

We first evaluate the inference accuracy and efficiency of ExpressGNN.

We compare our method with several strong MLN inference methods on UW-CSE, Cora and Kinship datasets.

We also conduct ablation study to explore the trade-off between GNN and tunable embeddings.

Experiment settings.

For the inference experiments, we fix the weights of all logic rules as 1.

A key advantage of MLN is that it can handle open-world setting in a consistent probabilistic framework.

Therefore, we adopt open-world setting for all the experiments, as opposed to closed-world setting where unobserved facts (except the query predicates) are assumed to be false.

We also report the performance under closed-world setting in Appendix C.

Prediction tasks.

The deductive logic inference task is to answer queries that typically involve single predicate.

For example in UW-CSE, the task is to predict the AdvisedBy(c,c ) relation for all persons in the set.

In Cora, the task is to de-duplicate entities, and one of the query predicates is SameAuthor(c,c ).

As for Kinship, the task is to predict whether a person is male or female, i.e., Male(c).

For each possible substitution of the query predicate with different entities, the model is tasked to predict whether it's true or not.

Inference accuracy.

The results of inference accuracy on three benchmark datasets are reported in Table 1 .

A hyphen in the entry indicates that it is either out of memory or exceeds the time limit (24 hours).

We denote our method as ExpressGNN-E since only the E-step is needed for the inference experiments.

Note that since the lifted BP is guaranteed to get identical results as BP (Singla & Domingos, 2008) , the results of these two methods are merged into one row.

For these experiments, ExpressGNN-E uses 64-dim GNN embeddings and 64-dim tunable embeddings.

On Cora, all the baseline methods fail to handle the data scale under open-world setting, and ExpressGNN-E achieves good inference accuracy.

On UW-CSE, ExpressGNN-E consistently outperforms all baselines.

The Kinship dataset is synthesized and noise-free, and the number of entities increases linearly on the five sets S1-S5.

HL-MRF achieves perfect accuracy for S1-S4, but is infeasible on the largest set S5.

ExpressGNN-E yields similar but not perfect results, which is presumably caused by the stochastic nature of our sampling and optimization procedure.

Inference efficiency.

The inference time corresponding to the experiments in Table 1 is summarized in Fig. 4 .

On UW-CSE (left table) , ExpressGNN-E uses much shorter time for inference compared to all the baseline methods, and meanwhile ExpressGNN-E achieves the best inference performance.

On Kinship (right figure) , as the data size grows linearly from S1 to S5, the inference time of most baseline methods grows exponentially, while ExpressGNN-E maintains a nearly constant time cost, demonstrating its nice scalability.

Some baseline methods such as MCMC and MC-SAT become infeasible for larger sets.

HL-MRF maintains a comparatively short inference time, however, it has a huge increase of memory cost and is not able to handle the largest set S5.

Ablation study.

ExpressGNN can trade-off the compactness and expressiveness of model by tuning the dimensionality of GNN and tunable embeddings.

We perform ablation study on the Cora dataset to investigate how this trade-off affects the inference accuracy.

Results of different configurations of ExpressGNN-E are shown in Table 2 .

It is observed that GNN64+Tune4 has comparable performance with Tune64, but is consistently better than GNN64.

Note that the number of parameters in GNN64+Tune4 is O(64 2 + 4|C|), while that in Tune64 is O(64|C|).

When the number of entities is large, GNN64+Tune4 has much less parameters to train.

This is consistent with our theoretical analysis result: As a compact model, GNN saves a lot of parameters, but GNN alone is not expressive enough.

A similar conclusion is observed for GNN64+Tune64 and Tune128.

Therefore, ExpressGNN seeks a combination of two types of embeddings to possess the advantage of both: having a compact model and being expressive.

The best configuration of their embedding sizes can be varied on different tasks, and determined by the goal: getting a portable model or better performance.

We evaluate ExpressGNN in the knowledge base completion task on the FB15K-237 dataset, and compare it with state-of-the-art knowledge base completion methods.

Experiment settings.

To generate logic rules, we use Neural LP (Yang et al., 2017) on the training set and pick up the candidates with top confidence scores.

See Appendix D for examples of selected logic rules.

We evaluate both inference-only and inference-and-learning version of ExpressGNN, denoted as ExpressGNN-E and ExpressGNN-EM, respectively.

Prediction task.

For each test query r(c, c ) with respect to relation r, the model is tasked to generate a rank list over all possible instantiations of r and sort them according to the model's confidence on how likely this instantiation is true.

Evaluation metrics.

Following existing studies (Bordes et al., 2013; Sun et al., 2019) , we use filtered ranking where the test triples are ranked against all the candidate triples not appearing in the dataset.

Candidate triples are generated by corrupting the subject or object of a query r(c, c ).

For evaluation, we compute the Mean Reciprocal Ranks (MRR), which is the average of the reciprocal rank of all the truth queries, and Hits@10, which is the percentage of truth queries that are ranked among the top 10.

Competitor methods.

Since none of the aforementioned MLN inference methods can scale up to this dataset, we compare ExpressGNN with a number of state-of-the-art methods for knowledge base completion, including Neural Tensor Network (NTN; Socher et al. (2013) ), Neural LP (Yang et al., 2017) , DistMult (Kadlec et al., 2017) , ComplEx (Trouillon et al., 2016) , TransE (Bordes et al., 2013) , RotatE (Sun et al., 2019) and pLogicNet .

The results of MLN and pLogicNet are directly taken from the paper .

For all the other baseline methods, we use publicly available code with the provided best hyperparameters to run the experiments.

Performance analysis.

The experimental results on the full training data are reported in Table 3 (100% columns).

Both ExpressGNN-E and ExpressGNN-EM significantly outperform all the baseline methods.

With learning the weights of logic rules, ExpressGNN-EM achieves the best performance.

Compared to MLN, ExpressGNN achieves much better performance since MLN only relies on the logic rules while ExpressGNN can also leverage the labeled data as additional supervision.

Compared to knowledge graph embedding methods such as TransE and RotatE, ExpressGNN can leverage the prior knowledge in logic rules and outperform these purely data-driven methods.

Data efficiency.

We investigate the data efficiency of ExpressGNN and compare it with baseline methods.

Following (Yang et al., 2017) , we split the knowledge base into facts / training / validation / testing sets, and vary the size of the training set from 0% to 100% to feed the model with complete facts set for training.

From Table 3 , we see that ExpressGNN performs significantly better than the baselines on smaller training data.

With more training data as supervision, data-driven baseline methods start to close the gap with ExpressGNN.

This clearly shows the benefit of leveraging the knowledge encoded in logic rules when there data is insufficient for supervised learning.

Zero-shot relational learning.

In practical scenarios, a large portion of the relations in the knowledge base are long-tail, i.e., most relations may have only a few facts (Xiong et al., 2018) .

Therefore, it is important to investigate the model performance on relations with insufficient training data.

We construct a zero-shot learning dataset based on FB15K-237 by forcing the training and testing data to have disjoint sets of relations.

Table 4 shows the results.

As expected, the performance of all the supervised relational learning methods drop to almost zero.

This shows the limitation of such methods when coping with sparse long-tail relations.

Neural LP is designed to handle new entities in the test set (Yang et al., 2017) , but still struggles to perform well in zero-shot learning.

In contrast, ExpressGNN leverages both the prior knowledge in logic rules and the neural relational embeddings for reasoning, which is much less affected by the scarcity of data on long-tail relations.

Both variants of our framework (ExpressGNN-E and ExpressGNN-EM) achieve significantly better performance.

This paper studies the probabilistic logic reasoning problem, and proposes ExpressGNN to combine the advantages of Markov Logic Networks in logic reasoning and graph neural networks in graph representation learning.

ExpressGNN addresses the scalability issue of Markov Logic Networks with efficient stochastic training in the variational EM framework.

ExpressGNN employs GNNs to capture the structure knowledge that is implicitly encoded in the knowledge graph, which serves as supplement to the knowledge from logic formulae.

ExpressGNN is a general framework that can trade-off the model compactness and expressiveness by tuning the dimensionality of the GNN and the embedding part.

Extensive experiments on multiple benchmark datasets demonstrates the effectiveness and efficiency of ExpressGNN.

We provide more examples in this section to show that it is more than a rare case that GNN embeddings alone are not expressive enough.

A.1 EXAMPLE 1 Now, we use another example in Fig. 7 to show that even when the local structures are the same, the posteriors can still be different, which is caused by the formulae.

...

??? (C,H,G,M) ... ??? B DATASET DETAILS

For our experiments, we use the following benchmark datasets:

??? The social network dataset UW-CSE (Richardson & Domingos, 2006) contains publicly available information of students and professors in the CSE department of UW.

The dataset is split into five sets according to the home department of the entities.

??? The entity resolution dataset Cora (Singla & Domingos, 2005) consists of a collection of citations to computer science research papers.

The dataset is also split into five subsets according to the field of research.

??? We introduce a synthetic dataset that resembles the popular Kinship dataset (Denham, 1973) .

The original dataset contains kinship relationships (e.g., Father, Brother) among family members in the Alyawarra tribe from Central Australia.

The synthetic dataset closely resembles the original Kinship dataset but with a controllable number of entities.

To generate a dataset with n entities, we randomly split n entities into two groups which represent the first and second generation respectively.

Within each group, entities are grouped into a few sub-groups representing the sisterand brother-hood.

Finally, entities from different sub-groups in the first generation are randomly coupled and a sub-group in the second generation is assigned to them as their children.

To generate the knowledge base, we traverse this family tree, and record all kinship relations for each entity.

We generate five kinship datasets (Kinship S1-S5) by linearly increasing the number of entities.

??? The knowledge base completion benchmark FB15K-237 (Toutanova & Chen, 2015 ) is a generic knowledge base constructed from Freebase (Bollacker et al., 2008) , which is designed to a more challenging variant of FB15K.

More specifically, FB15K-237 is constructed by removing nearduplicate and inverse relations from FB15K.

The dataset is split into training / validation / testing and we use the same split of facts from training as in prior work (Yang et al., 2017) .

The complete statistics of these datasets are shown in Table 5 .

Examples of logic formulae used in four benchmark datasets are listed in Table 7 .

In Sec. 6.1 we compare ExpressGNN with five probabilistic inference methods under open-world semantics.

This is different from the original works, where they generally adopt the closed-world setting due to the scalability issues.

More specifically, the original works assume that the predicates (except the ones in the query) observed in the knowledge base is closed, meaning for all instantiations For sanity checking, we also conduct these experiments with a closed-world setting.

We found the results summarized in Table 6 are close to those reported in the original works.

This shows that we have a fair setup (including memory size, hyperparameters, etc.) for those competitor methods.

Additionally, one can find that the AUC-PR scores compared to those (Table 1) under open-world setting are actually better.

This is due to the way the datasets were originally collected and evaluated generally complies with the closed-world assumption.

But this is very unlikely to be true for realworld and large-scale knowledge base such as Freebase and WordNet, where many true facts between entities are not observed.

Therefore, in general, the open-world setting is much more reasonable, which we follow throughout this paper.

We list some examples of logic formulae used in four benchmark datasets in Table 7 .

The full list of logic formulae is available in our source code repository.

Note that these formulae are not necessarily as clean as being always true, but are typically true.

For UW-CSE and Cora, we use the logic formulae provided in the original dataset.

UW-CSE provides 94 hand-coded logic formulae, and Cora provides 46 hand-coded rules.

For Kinship, we hand-code 22 first-order logic formulae.

For FB15K-237, we first use Neural LP (Yang et al., 2017) on the full data to generate candidate rules.

Then we select the ones that have confidence scores higher than 90% of the highest scored formulae sharing the same target predicate.

We also de-duplicate redundant rules that can be reduced to other rules by switching the logic variables.

Finally, we have generated 509 logic formulae for FB15K-237.

Theorem.

Consider a knowledge base K = (C, R, O) and any r ??? R. Two latent random variables X := r(c 1 , . . .

, c n ) and X := r(c 1 , . . . , c n ) have the same posterior distribution in any MLN if and only if (c 1 , ?? ?? ?? , c n )

Then we give the proof as follows.

A logic formula f can be represented as a factor graph, G f = (C f , R f , E f ), where nodes on one side of the graph is the set of distinct constants C f needed in the formula, while nodes on the other side is the set of predicates R f used to define the formula.

The set of edges, E f , will connect constants to predicates or predicate negation.

That is, an edge e = (c, r, i) between node c and predicate r exists, if the predicate r use constant c in its i-th argument.

We note that the set of distinctive constants used in the definition of logic formula are templates where actual constant can be instantiated from C. An illustration of logic formula factor graph can be found in Fig. 8 .

Similar to the factor graph for the knowledge base, we also differentiate the type of edges by the position of the argument.

Therefore, every single formula can be represented by a factor graph.

We will construct a factor graph representation to define a particular formula, and show that the MLN induced by this formula will result in different posteriors for r(c 1 , . . .

, c n ) and r(c 1 , . . . , c n ).

The factor graph for the formula is constructed in the following way (see Fig. 7 as an example of the resulting formula constructed using the following steps): The proof of this claim is given at the end of this proof.

(ii) Next, we use G * c1:n to define a formula f .

We first initialize the definition of the formula value as f (c 1 , . . .

, c n ,c 1 , . . . ,c n ) = ??? r(ar) :r(ar) ??? G * c1:n ??? r(c 1 , . . .

, c n ).

Then, we changer(ar) in this formula to the negation ??r(ar) if the observed value ofr(ar) is 0 in G * c1:n .

We have defined a formula f using the above two steps.

Suppose the MLN only contains this formula f .

Then the two nodes r(c 1 , . . .

, c n ) and r(c 1 , . . . , c n ) in this MLN must be distinguishable.

The reason is, in MLN, r(c 1 , . . .

, c n ) is connected to a ground formula f (c 1 , . . .

, c n ,c 1 , . . .

,c n ), whose factor graph representation is G * c1:n ??? r(c 1 , . . . , c n ).

In this formula, all variables are observed in the knowledge base K except for r(c 1 , . . .

, c n ) and and the observation set is O * c .

The formula value is f (c 1 , . . . , c n ,c 1 , . . . ,c n ) = (1 ??? r(c 1 , . . . , c n )) .

Clarification: Eq. 10 is used to define a formula and c i in this equation can be replaced by other constants, while Eq. 11 represents a ground formula whose arguments are exactly c 1 , . . .

, c n ,c 1 , . . .

,c n .

Based on (Condition), there is NO formula f (c 1 , . . .

, c n ,c 1 , . . .

,c n ) that contains r(c 1 , . . .

, c n ) has

<|TLDR|>

@highlight

We employ graph neural networks in the variational EM framework for efficient inference and learning of Markov Logic Networks.